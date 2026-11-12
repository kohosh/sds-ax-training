"""Day 4 — 운영 지원 MCP 서버와 장애 복구.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. 도구는 예외를 밖으로 던지지 않습니다. 무엇이 잘못됐는지 설명하는 문자열을 돌려주세요.
     그 문자열은 모델이 읽고 다음 행동을 정하는 프롬프트입니다.
  3. 자격증명은 절대 반환값이나 로그에 남기지 마세요.

[이 파일의 구성]
  - 기초 파트: 운영 지원 도구 3개(search_runbook / get_service_owner / query_recent_incidents),
    MCP 로 내보낼 명세(TOOL_SPECS)와 구현 매핑(TOOLS_IMPL),
    재시도 + 대체 경로를 갖춘 자산 조회(get_asset_status / get_asset_status_backup).
  - 심화 파트: classify_failure 로 실패를 retryable / backoff / fatal 로 가르고,
    get_asset_status 가 그 판정에 따라 재시도 전략을 바꿉니다.
  - 공통 안전장치: 모든 반환 문자열과 로그는 _redact()를 거쳐 자격증명을 가립니다.

[모듈 최상단 규칙]
  모듈을 import 하는 것만으로는 모델·서버·네트워크 자원을 만들지 않습니다.
  JSON 데이터 파일을 읽는 것 외에는 모두 함수 안에서 일어납니다.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

DATA = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "ops_assets.json").read_text(
        encoding="utf-8"
    )
)

# 교육용 더미 자격증명입니다. 이 값이 로그나 반환값에 남으면 채점에서 감점됩니다.
BACKEND_TOKEN = "sk-internal-dummy-1234"

# 백엔드 "총 호출" 상한입니다. (최초 1회 + 재시도 2회 = 3회)
# 채점 기준: 타임아웃이면 2회 이상 MAX_RETRIES 회 이하로 호출해야 합니다.
MAX_RETRIES = 3


# ══════════════════════════════════════════════════════════════════
# 공통 안전장치: 자격증명 가리기 + 로거
# ══════════════════════════════════════════════════════════════════

_MASK = "[REDACTED]"

# 값 자체를 몰라도 "생김새"로 잡아내는 패턴들입니다.
# 순서가 중요합니다. 'Authorization: Bearer xxx' 처럼 겹치는 경우 앞 패턴이 먼저 처리됩니다.
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    # Authorization 헤더 / Bearer 토큰: "Bearer sk-..." -> "Bearer [REDACTED]"
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=\-]+"),
    # sk-로 시작하는 API 키 형태
    re.compile(r"\bsk-[A-Za-z0-9_\-]{4,}"),
    # key=value / key: value 형태 (token=..., api_key: ..., password=...)
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|passwd|authorization)"
        r"(\s*[=:]\s*)(?!\[REDACTED\]|(?i:bearer)\s)['\"]?[^\s'\",;]+"
    ),
]

# 응답 dict 에 이런 이름의 키가 있으면 값 자체를 통째로 가립니다.
_SENSITIVE_KEYS = re.compile(
    r"(?i)(token|secret|password|passwd|api[_-]?key|authorization|credential|cookie)"
)


def _redact(text: Any) -> str:
    """문자열에서 자격증명으로 보이는 부분을 모두 가립니다.

    1) 알려진 값(BACKEND_TOKEN)은 정확히 일치하는 부분을 지웁니다.
    2) 값을 모르는 비밀값은 생김새(Bearer, sk-, key=value)로 지웁니다.
    반환 문자열과 로그 메시지는 반드시 이 함수를 거친 뒤에 밖으로 나갑니다.
    """
    s = text if isinstance(text, str) else str(text)
    if BACKEND_TOKEN:
        s = s.replace(BACKEND_TOKEN, _MASK)
    s = _SECRET_PATTERNS[0].sub(lambda m: f"{m.group(1)} {_MASK}", s)
    s = _SECRET_PATTERNS[1].sub(_MASK, s)
    s = _SECRET_PATTERNS[2].sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASK}", s)
    return s


def _scrub(value: Any) -> Any:
    """dict/list 응답을 재귀적으로 돌며 민감한 키의 값을 가립니다."""
    if isinstance(value, dict):
        return {
            k: (_MASK if _SENSITIVE_KEYS.search(str(k)) else _scrub(v)) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


class _RedactingFilter(logging.Filter):
    """로그 레코드가 핸들러로 가기 전에 메시지에서 자격증명을 지웁니다.

    logger.warning("... %s", exc) 처럼 인자로 넘어온 값도 포매팅한 뒤 지우므로,
    예외 메시지 안에 토큰이 섞여 들어와도 로그에는 남지 않습니다.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            record.msg = _redact(record.getMessage())
            record.args = ()
        except Exception:  # noqa: BLE001 - 로깅 때문에 도구가 죽으면 안 됩니다.
            record.msg, record.args = "(로그 메시지를 안전하게 만들지 못해 생략함)", ()
        return True


logger = logging.getLogger("day04.ops_tools")
if not any(isinstance(f, _RedactingFilter) for f in logger.filters):
    logger.addFilter(_RedactingFilter())


def _clean_arg(value: Any) -> str:
    """도구 인자를 문자열로 정리합니다. None/숫자 등이 와도 예외 없이 처리합니다."""
    if value is None:
        return ""
    return str(value).strip()


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# 모델이 "500 에러", "메모리 부족", "인증서 만료"처럼 자연어로 물어도
# 런북 키(5xx / oom / cert)로 이어지도록 동의어를 둡니다.
_RUNBOOK_ALIASES: dict[str, tuple[str, ...]] = {
    "5xx": ("500", "502", "503", "504", "server error", "서버 에러", "서버 오류", "5백"),
    "oom": ("out of memory", "outofmemory", "memory", "메모리", "힙", "heap"),
    "cert": ("certificate", "ssl", "tls", "인증서", "x509"),
}


def search_runbook(symptom: str) -> str:
    """증상 키워드로 대응 런북을 찾습니다."""
    try:
        runbooks: dict[str, str] = DATA["runbooks"]
        available = ", ".join(runbooks)
        query = _clean_arg(symptom)

        if not query:
            return (
                "에러: symptom 인자가 비어 있습니다. "
                f"증상 키워드 하나를 넣어 다시 호출하세요. 쓸 수 있는 값: {available}"
            )

        key = query.lower()
        matched: str | None = key if key in runbooks else None

        # 1) 동의어 / 2) 문장 속 키워드 포함("5xx 급증", "OOM 발생") 순서로 찾습니다.
        if matched is None:
            for rb_key, aliases in _RUNBOOK_ALIASES.items():
                if rb_key in runbooks and any(a in key for a in aliases):
                    matched = rb_key
                    break
        if matched is None:
            matched = next((k for k in runbooks if k in key), None)

        if matched is None:
            return (
                f"'{_redact(query)}' 는 등록되지 않은 증상 키워드입니다. "
                f"쓸 수 있는 값: {available}. "
                "이 중 가장 가까운 키워드로 다시 호출하세요. "
                "담당자가 필요하면 get_service_owner, 과거 사례가 필요하면 "
                "query_recent_incidents 를 쓰세요."
            )

        note = "" if matched == key else f" (입력 '{_redact(query)}' 를 '{matched}' 로 해석함)"
        return f"[런북: {matched}]{note}\n{runbooks[matched]}"
    except Exception as exc:  # noqa: BLE001 - 도구는 예외를 밖으로 던지지 않습니다.
        logger.exception("search_runbook 내부 오류: %s", exc)
        return f"에러: 런북 조회 중 내부 오류가 발생했습니다 ({type(exc).__name__}). 잠시 후 다시 시도하세요."


def _find_service(service: str, table: dict[str, Any]) -> str | None:
    """서비스 이름을 대소문자/공백 차이에 관계없이 찾습니다."""
    key = _clean_arg(service).lower().replace(" ", "-").replace("_", "-")
    if key in table:
        return key
    return next((k for k in table if k.lower() == key), None)


def _known_services() -> list[str]:
    """owners 와 incidents 에 등장하는 서비스 이름 전체(중복 제거, 정렬)."""
    return sorted(set(DATA["owners"]) | set(DATA["incidents"]))


def get_service_owner(service: str) -> str:
    """서비스의 담당팀과 온콜 담당자를 조회합니다."""
    try:
        owners: dict[str, dict] = DATA["owners"]
        query = _clean_arg(service)
        available = ", ".join(owners)

        if not query:
            return (
                "에러: service 인자가 비어 있습니다. "
                f"서비스 이름을 넣어 다시 호출하세요. 등록된 서비스: {available}"
            )

        name = _find_service(query, owners)
        if name is None:
            return (
                f"'{_redact(query)}' 서비스의 담당 정보를 찾을 수 없습니다. "
                f"등록된 서비스: {available}. 정확한 서비스 이름으로 다시 호출하세요. "
                "자산 ID(SRV-001 등)를 알고 있다면 get_asset_status 로 owner 팀을 먼저 확인할 수 있습니다."
            )

        o = owners[name]
        return (
            f"[담당 정보] {name}\n"
            f"- 담당팀: {o.get('team', '미등록')}\n"
            f"- 온콜: {o.get('oncall', '미등록')}\n"
            f"- 알림 채널: {o.get('slack', '미등록')}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_service_owner 내부 오류: %s", exc)
        return f"에러: 담당 정보 조회 중 내부 오류가 발생했습니다 ({type(exc).__name__}). 잠시 후 다시 시도하세요."


def query_recent_incidents(service: str) -> str:
    """서비스의 최근 인시던트 이력을 조회합니다."""
    try:
        incidents: dict[str, list[dict]] = DATA["incidents"]
        query = _clean_arg(service)
        known = _known_services()

        if not query:
            return (
                "에러: service 인자가 비어 있습니다. "
                f"서비스 이름을 넣어 다시 호출하세요. 등록된 서비스: {', '.join(known)}"
            )

        name = _find_service(query, {k: None for k in known})
        if name is None:
            return (
                f"'{_redact(query)}' 는 등록되지 않은 서비스입니다. "
                f"등록된 서비스: {', '.join(known)}. 정확한 서비스 이름으로 다시 호출하세요."
            )

        rows = sorted(incidents.get(name, []), key=lambda r: r.get("at", ""), reverse=True)
        if not rows:
            # '없는 서비스'와 '이력이 없는 서비스'는 다른 답입니다. 모델이 헷갈리지 않게 구분합니다.
            return f"[최근 인시던트] {name}: 기록된 인시던트가 없습니다 (서비스는 등록되어 있음)."

        lines = [f"[최근 인시던트] {name}: {len(rows)}건 (최신순)"]
        for r in rows:
            lines.append(
                f"- {r.get('at', '?')} {r.get('id', '?')} {r.get('title', '')} "
                f"(원인: {r.get('cause', '미상')})"
            )
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.exception("query_recent_incidents 내부 오류: %s", exc)
        return f"에러: 인시던트 조회 중 내부 오류가 발생했습니다 ({type(exc).__name__}). 잠시 후 다시 시도하세요."


# MCP로 노출할 도구 명세입니다.
#   이 설명문은 우리 코드 밖(다른 팀 클라이언트, 다른 모델)에서 읽힙니다.
#   그래서 '무엇을 하는지' + '언제 쓰는지' + '언제 쓰지 말아야 하는지(대신 쓸 도구)'를 모두 적습니다.
#   인자 설명에는 허용 값 예시를 넣어 모델이 인자 값을 지어내지 않게 합니다.
TOOL_SPECS: list[dict] = [
    {
        "name": "search_runbook",
        "description": (
            "증상 키워드로 장애 대응 런북(조치 절차)을 찾습니다. "
            "에러 급증, 메모리 부족, 인증서 문제처럼 '지금 무엇부터 조치해야 하는지'가 필요할 때 사용합니다. "
            "담당자·연락처를 찾을 때는 get_service_owner, 과거 장애 이력을 볼 때는 "
            "query_recent_incidents 를 쓰고 이 도구는 쓰지 않습니다. "
            "등록되지 않은 키워드면 사용 가능한 키워드 목록을 돌려줍니다."
        ),
        "args": {
            "symptom": "증상 키워드 한 개. 허용 값: 5xx, oom, cert (예: '5xx'). "
            "'500 에러', '메모리 부족', '인증서 만료' 같은 표현도 가장 가까운 키워드로 해석합니다.",
        },
    },
    {
        "name": "get_service_owner",
        "description": (
            "서비스 이름으로 담당팀, 현재 온콜 담당자, 알림 Slack 채널을 조회합니다. "
            "장애를 누구에게 에스컬레이션할지, 어느 채널에 알릴지 정할 때 사용합니다. "
            "조치 절차가 필요하면 search_runbook, 과거 장애 이력은 query_recent_incidents 를 쓰세요. "
            "자산 ID(SRV-xxx)로는 조회되지 않으니 서비스 이름을 넣어야 합니다."
        ),
        "args": {
            "service": "서비스 이름(소문자, 하이픈 구분). 예: payment-api, auth-service, gateway",
        },
    },
    {
        "name": "query_recent_incidents",
        "description": (
            "서비스 이름으로 최근 인시던트 이력(ID, 일자, 제목, 원인)을 최신순으로 조회합니다. "
            "비슷한 장애가 전에 있었는지, 반복 원인이 무엇인지 확인할 때 사용합니다. "
            "현재 실시간 상태 확인용이 아니며(자산 상태는 get_asset_status), "
            "담당자 조회에는 get_service_owner 를 쓰세요. 이력이 없으면 '기록 없음'을 돌려줍니다."
        ),
        "args": {
            "service": "서비스 이름(소문자, 하이픈 구분). 예: payment-api, auth-service, gateway",
        },
    },
]

# 이름 -> 실제 함수 매핑. TOOL_SPECS의 name과 키가 정확히 일치합니다.
TOOLS_IMPL: dict[str, Any] = {
    "search_runbook": search_runbook,
    "get_service_owner": get_service_owner,
    "query_recent_incidents": query_recent_incidents,
}


def _default_fetch(asset_id: str) -> dict:
    """기본 자산 조회 경로입니다. 실제로는 외부 시스템을 호출한다고 가정합니다."""
    asset = DATA["assets"].get(asset_id)
    if not asset:
        raise LookupError(f"asset not found: {asset_id}")
    return asset


# 재시도 간격(초). 짧게 두되, 테스트에서 monkeypatch 로 _sleep 을 바꿔 끼울 수 있게 분리합니다.
RETRY_DELAY_SEC = 0.05     # retryable: 짧고 일정한 간격
BACKOFF_BASE_SEC = 0.2     # backoff : 0.2 -> 0.4 -> 0.8 ... (지수 증가)
BACKOFF_MAX_SEC = 2.0      # backoff 한 번의 최대 대기
_sleep: Callable[[float], None] = time.sleep

_REQUIRED_FIELDS = ("status",)


def _validate_response(resp: Any) -> dict:
    """백엔드 응답 스키마를 검사합니다. 어긋나면 ValueError(= fatal)로 바꿉니다.

    README 로그의 'field status missing' 상황을 여기서 잡아냅니다.
    스키마 오류는 다시 물어도 같으므로 classify_failure 가 fatal 로 분류하고 즉시 대체 경로로 갑니다.
    """
    if not isinstance(resp, dict):
        raise ValueError(f"schema validation failed: expected dict, got {type(resp).__name__}")
    missing = [f for f in _REQUIRED_FIELDS if f not in resp or resp[f] in (None, "")]
    if missing:
        raise ValueError(f"schema validation failed: field {missing[0]!r} missing")
    return resp


def _format_asset(asset_id: str, asset: dict) -> str:
    """자산 dict 를 '키=값' 한 줄로. 민감한 키는 가리고, status 를 맨 앞에 둡니다."""
    safe = _scrub(asset)
    order = ["status", "name", "owner"]
    keys = [k for k in order if k in safe] + [k for k in safe if k not in order]
    return f"{asset_id} | " + " | ".join(f"{k}={safe[k]}" for k in keys)


def _delay_for(kind: str, attempt: int) -> float:
    """분류와 시도 횟수로 다음 재시도까지 기다릴 시간을 정합니다."""
    if kind == "backoff":
        return min(BACKOFF_BASE_SEC * (2 ** (attempt - 1)), BACKOFF_MAX_SEC)
    return RETRY_DELAY_SEC


def get_asset_status(asset_id: str, fetch: Callable[[str], dict] | None = None) -> str:
    """자산 상태를 조회합니다. 실패하면 재시도하고, 그래도 안 되면 대체 경로를 씁니다.

    반드시 지킬 것:
      - 어떤 경우에도 예외를 밖으로 던지지 않습니다.
      - 대체 경로로 넘어갔다면 반환 문자열에 'get_asset_status_backup' 이름을 남깁니다.
        모델이 다음에 무엇을 할지 알아야 하기 때문입니다.
      - BACKEND_TOKEN 값을 반환 문자열에 넣지 마세요.

    동작 (심화: classify_failure 판정에 따라 갈림):
      - retryable (타임아웃/연결 오류): 짧은 간격으로 재시도. 총 호출 MAX_RETRIES 회 이하.
      - backoff   (rate limit)       : 간격을 지수적으로 늘려 재시도. 총 호출 MAX_RETRIES 회 이하.
      - fatal     (스키마/인증/없는 자원): 재시도 없이 즉시 get_asset_status_backup 으로.
      매 시도마다 다시 분류하므로 '타임아웃 -> 스키마 오류'처럼 실패 종류가 바뀌어도 바로 반응합니다.

    반환 문자열에는 답이 어느 경로에서 나왔는지(기본 조회 / 대체 경로)와 시도 횟수를 남깁니다.

    Args:
        asset_id: 조회할 자산 ID
        fetch: 외부 조회 함수. 채점기가 실패하는 함수를 넣습니다. None이면 기본 경로.
    """
    try:
        fetch = fetch or _default_fetch
        aid = _clean_arg(asset_id).upper()

        # 0) 입력 검증: 빈 ID 로 백엔드를 두드리지 않습니다.
        if not aid:
            return (
                "에러: asset_id 가 비어 있습니다. 'SRV-001' 형식의 자산 ID 를 넣어 다시 호출하세요. "
                f"등록된 자산 예: {', '.join(list(DATA['assets'])[:5])}"
            )

        attempts = 0
        history: list[str] = []        # 시도별 실패 요약 (모델/운영자가 읽는 용도, 이미 redact됨)
        last_kind = "fatal"

        # 1) 기본 경로 호출 + 분류 기반 재시도
        while attempts < MAX_RETRIES:
            attempts += 1
            try:
                resp = _validate_response(fetch(aid))
            except Exception as exc:  # noqa: BLE001 - 어떤 실패든 분류 후 처리합니다.
                last_kind = classify_failure(exc)
                summary = _redact(f"{type(exc).__name__}: {exc}")[:200]
                history.append(f"{attempts}회차 {last_kind} ({summary})")
                logger.warning(
                    "get_asset_status(%s) %d/%d회차 실패 [%s] %s",
                    aid, attempts, MAX_RETRIES, last_kind, summary,
                )
                if last_kind == "fatal" or attempts >= MAX_RETRIES:
                    break
                _sleep(_delay_for(last_kind, attempts))
                continue

            # 성공: 재시도 끝에 살아난 값이면 그 사실도 남깁니다(폴백 문구로 덮어쓰지 않음).
            via = "첫 시도" if attempts == 1 else f"{attempts}회차 재시도에서 복구"
            logger.info("get_asset_status(%s) 성공 (%s)", aid, via)
            out = (
                f"[자산 상태 · 기본 조회 경로] {_format_asset(aid, resp)}\n"
                f"(출처: get_asset_status 실시간 조회, {via}, 총 {attempts}회 호출)"
            )
            if history:
                out += "\n이전 실패: " + "; ".join(history)
            return _redact(out)

        # 2) 기본 경로 포기 -> 대체 경로
        reason = {
            "fatal": "다시 물어도 결과가 같은 오류(스키마/인증/없는 자원)라 재시도하지 않았습니다",
            "retryable": f"일시적 오류가 {attempts}회 연속 발생해 재시도 상한(MAX_RETRIES={MAX_RETRIES})에 도달했습니다",
            "backoff": f"요청 한도 초과가 간격을 늘린 {attempts}회 시도 후에도 풀리지 않았습니다",
        }.get(last_kind, "기본 조회에 실패했습니다")
        logger.warning(
            "get_asset_status(%s) 대체 경로 전환: get_asset_status_backup (분류=%s, 시도=%d)",
            aid, last_kind, attempts,
        )
        backup = get_asset_status_backup(aid)
        out = (
            f"[대체 경로 사용: get_asset_status_backup] 기본 조회 실패 -> 대체 경로로 전환했습니다.\n"
            f"- 실패 분류: {last_kind} / 기본 경로 호출 {attempts}회. {reason}.\n"
            f"- 실패 이력: {'; '.join(history)}\n"
            f"- get_asset_status_backup 결과: {backup}\n"
            "주의: 대체 경로 값은 캐시일 수 있어 '현재 상태'로 단정해 보고하지 마세요. "
            "보고할 때 출처가 get_asset_status_backup 임을 밝히고, "
            + (
                "잠시 후 get_asset_status 로 다시 확인하세요."
                if last_kind != "fatal"
                else "같은 인자로 get_asset_status 를 바로 다시 부르지 마세요(결과가 같습니다). "
                "자산 ID 가 맞는지 확인하거나 담당팀(get_service_owner)에 문의하세요."
            )
        )
        return _redact(out)
    except Exception as exc:  # noqa: BLE001 - 최후 방어선: 어떤 경우에도 예외를 밖으로 던지지 않습니다.
        logger.exception("get_asset_status 내부 오류: %s", exc)
        return _redact(
            f"에러: 자산 상태 조회 중 내부 오류가 발생했습니다 ({type(exc).__name__}). "
            "get_asset_status_backup 으로 캐시 값을 확인하거나 잠시 후 다시 시도하세요."
        )


def get_asset_status_backup(asset_id: str) -> str:
    """대체 조회 경로입니다. 캐시된 값이나 축약 정보를 돌려준다고 가정합니다."""
    try:
        aid = _clean_arg(asset_id).upper()
        assets: dict[str, dict] = DATA.get("assets", {})
        if not aid:
            return (
                "[get_asset_status_backup] 에러: asset_id 가 비어 있습니다. "
                f"등록된 자산: {', '.join(assets)}"
            )
        cached = assets.get(aid)
        if not cached:
            return (
                f"[get_asset_status_backup] '{_redact(aid)}' 자산은 캐시에도 없습니다. "
                f"등록된 자산: {', '.join(assets)}. 자산 ID 를 확인한 뒤 다시 호출하세요."
            )
        return _redact(
            f"[get_asset_status_backup · 캐시] {_format_asset(aid, cached)} "
            "(실시간 값 아님: 마지막으로 캐시된 상태)"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_asset_status_backup 내부 오류: %s", exc)
        return (
            f"[get_asset_status_backup] 에러: 대체 경로 조회도 실패했습니다 ({type(exc).__name__}). "
            "담당팀에 직접 문의하세요."
        )


# 자산 조회 도구도 MCP 로 내보낼 수 있게 별도 명세를 둡니다.
# (TOOL_SPECS 는 과제가 요구한 운영 도구 3개만 담고, mcp_server.py 가 두 목록을 함께 등록합니다.)
# MCP 로 노출할 때는 fetch 인자를 숨기고 asset_id 만 받습니다. fetch 는 코드 안에서만 꽂는 값입니다.
ASSET_TOOL_SPECS: list[dict] = [
    {
        "name": "get_asset_status",
        "description": (
            "자산 ID 로 서버 자산의 현재 상태(status), 이름, 담당팀을 실시간 조회합니다. "
            "장애 조사 중 특정 서버가 지금 running/degraded 인지 확인할 때 사용합니다. "
            "기본 조회가 실패하면 자동으로 재시도하고, 그래도 안 되면 get_asset_status_backup(캐시)으로 "
            "넘어가며 그 사실을 결과에 밝힙니다. 서비스 이름으로는 조회되지 않습니다."
        ),
        "args": {"asset_id": "자산 ID. 'SRV-' + 숫자 3자리 형식. 예: SRV-001, SRV-002"},
    },
    {
        "name": "get_asset_status_backup",
        "description": (
            "자산 상태의 캐시 값을 조회하는 대체 경로입니다. 실시간 값이 아닙니다. "
            "get_asset_status 가 실패했거나 기본 백엔드가 장애일 때만 사용하고, "
            "평소 상태 확인에는 get_asset_status 를 쓰세요. 결과를 보고할 때는 캐시 값임을 밝히세요."
        ),
        "args": {"asset_id": "자산 ID. 'SRV-' + 숫자 3자리 형식. 예: SRV-001"},
    },
]

ASSET_TOOLS_IMPL: dict[str, Any] = {
    "get_asset_status": lambda asset_id: get_asset_status(asset_id),
    "get_asset_status_backup": get_asset_status_backup,
}


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 판단 기준은 하나: "다시 물었을 때 답이 달라질 수 있는가?"
#   - 달라질 수 있다            -> retryable (짧게 재시도)
#   - 달라지지만 천천히 물어야 함 -> backoff   (간격 늘려 재시도)
#   - 몇 번을 물어도 같다        -> fatal     (즉시 대체 경로)

_BACKOFF_MSG = re.compile(
    r"(?i)(\b429\b|rate[\s_-]?limit|too many requests|throttl|quota (exceeded|exhausted)"
    r"|slow down|요청 한도|호출 한도|한도 초과)"
)
_FATAL_MSG = re.compile(
    r"(?i)(schema|validation|malformed|missing|invalid (json|response|payload|format)"
    r"|\b40[0134]\b|\b422\b|unauthori[sz]ed|unauthenticated|forbidden|authentication"
    r"|auth(entication|orization)? failed|invalid (token|credential|api key)|expired token"
    r"|permission denied|access denied|not found|no such|does not exist|unknown asset"
    r"|스키마|검증 실패|인증 실패|권한 없음|찾을 수 없)"
)
_RETRYABLE_MSG = re.compile(
    r"(?i)(time[\s_-]?out|timed out|deadline exceeded|connection (reset|refused|aborted|error|closed)"
    r"|reset by peer|broken pipe|temporar(il)?y|unavailable|\b50[0234]\b|bad gateway"
    r"|network (error|unreachable)|try again|eof occurred|시간 초과|일시적|연결 (오류|끊김|실패))"
)

# 예외 타입 이름(대소문자 무시)에 이런 조각이 있으면 판정에 씁니다. requests/httpx/boto 의 예외 이름도 잡힙니다.
_BACKOFF_TYPES = ("ratelimit", "throttl", "toomanyrequests", "quotaexceeded")
_FATAL_TYPES = ("schema", "validation", "auth", "permission", "forbidden", "notfound", "decode")
_RETRYABLE_TYPES = ("timeout", "connection", "temporar", "unavailable", "network")


def classify_failure(error: BaseException | str) -> str:
    """실패를 세 종류로 분류합니다. LLM을 호출하지 않습니다.

    반환값은 정확히 다음 셋 중 하나여야 합니다.

      "retryable"  타임아웃, 일시적 연결 오류      -> 짧게 재시도
      "backoff"    요청 한도 초과 (rate limit)     -> 간격을 늘려 재시도
      "fatal"      스키마 오류, 인증 실패, 없는 자원 -> 재시도하지 말고 즉시 대체 경로

    판단 근거는 예외 타입이나 메시지 문자열 어느 쪽을 써도 됩니다.

    판정 순서 (앞에서 걸리면 확정):
      1) rate limit 신호(메시지/타입/HTTP 429) -> backoff
         한도 초과는 RuntimeError 등 어떤 타입으로도 올 수 있어 가장 먼저 봅니다.
      2) 예외 타입: 타임아웃/연결 계열 -> retryable,
                   LookupError(KeyError)/ValueError/TypeError/PermissionError 계열 -> fatal
      3) 메시지 키워드: fatal 신호 -> fatal, 일시적 신호 -> retryable
      4) 어느 것에도 해당하지 않으면 fatal.
         정체를 모르는 실패를 무작정 재시도하면 README 로그의 39회 재시도가 재현되므로,
         모를 때는 재시도하지 않고 대체 경로로 넘어가는 쪽을 택합니다.
    """
    try:
        if isinstance(error, type) and issubclass(error, BaseException):
            exc_type: type | None = error
            message = ""
        elif isinstance(error, BaseException):
            exc_type = type(error)
            message = str(error)
        else:
            exc_type = None
            message = _clean_arg(error)

        # HTTP 상태 코드를 속성으로 들고 있는 예외(requests/httpx 스타일)도 봅니다.
        status = None
        if isinstance(error, BaseException):
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            resp = getattr(error, "response", None)
            if status is None and resp is not None:
                status = getattr(resp, "status_code", None)
        if isinstance(status, int):
            message = f"{status} {message}"

        type_names = (
            " ".join(t.__name__.lower() for t in exc_type.__mro__) if exc_type is not None else ""
        )

        # 1) rate limit
        if _BACKOFF_MSG.search(message) or any(t in type_names for t in _BACKOFF_TYPES):
            return "backoff"

        # 2) 예외 타입
        if exc_type is not None:
            if issubclass(exc_type, (TimeoutError, ConnectionError)):
                return "retryable"
            if issubclass(exc_type, (LookupError, ValueError, TypeError, PermissionError)):
                return "fatal"
            own_name = exc_type.__name__.lower()
            if any(t in own_name for t in _FATAL_TYPES):
                return "fatal"
            if any(t in own_name for t in _RETRYABLE_TYPES):
                return "retryable"

        # 3) 메시지 키워드
        if _FATAL_MSG.search(message):
            return "fatal"
        if _RETRYABLE_MSG.search(message):
            return "retryable"

        # 4) 모르는 실패
        return "fatal"
    except Exception:  # noqa: BLE001 - 분류기가 터져도 호출부는 안전하게 대체 경로로 갑니다.
        return "fatal"
