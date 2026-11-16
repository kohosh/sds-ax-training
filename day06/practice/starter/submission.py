"""Day 6 — Supervisor 인시던트 대응팀.

전문 Agent 셋을 Supervisor가 지휘합니다.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. `route_question`은 LLM을 호출하지 않습니다. 결정적 규칙으로 배분하세요.
     실제 운영에서는 모델이 배분하지만, 규칙으로 먼저 나눠 보면
     '무엇을 근거로 나누는가'가 분명해집니다.

[이 파일의 구성]
  ┌──────────── langgraph_supervisor.create_supervisor 로 조립 ────────────┐
  │                                                                          │
  │   START ─▶ supervisor ─┬─▶ log_agent     (get_logs, search_error_logs)  │
  │              ▲   │     ├─▶ metric_agent  (get_metrics, get_deploy_history)
  │              │   │     └─▶ runbook_agent (search_runbook, get_service_owner,
  │              │   ▼                         query_recent_incidents)      │
  │              │  END       워커 ──(handoff back)──▶ supervisor            │
  └──────────────┴───────────────────────────────────────────────────────────┘

  supervisor 노드 내부 (create_react_agent) 흐름:
    pre_model_hook (_supervisor_gate)
        1) route_question 으로 targets 계산 → state["targets"] 에 기록
        2) 직전 워커 산출물을 judge_output 으로 판정 → 억제된 것은 사유와 함께 교체
           → state["gate_log"] 에 판정 기록
    model (_RuleRoutedSupervisorModel)
        - 아직 안 부른 target 이 남아 있으면: LLM 없이 transfer_to_<agent> 도구 호출 생성
          (target 이 둘 이상이면 AGENT_NAMES 순서로 한 명씩 차례로 handoff)
        - 모두 끝났으면: 게이트를 통과한 보고만 모아 LLM 한 번으로 종합 답변
        - target 이 없으면(무관한 질문): Supervisor 가 직접 답변

  즉 "누구에게 보낼지"는 코드(규칙)가, "보고를 어떻게 종합할지"만 모델이 정합니다.
  오답이 나면 라우팅(결정적)과 워커(확률적) 중 어느 층이 틀렸는지 바로 가릴 수 있습니다.

[모듈 최상단 규칙]
  import 만으로는 모델·네트워크 자원을 만들지 않습니다. 모델은 build_supervisor 안에서 만듭니다.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"


def _default_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


def get_text(message: Any) -> str:
    """메시지(또는 문자열)에서 텍스트만 뽑습니다.

    ChatBedrockConverse 는 content 를 블록 리스트로 주기도 하므로 text 블록만 모아 이어 붙입니다.
    """
    content = getattr(message, "content", message)
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


# ══════════════════════════════════════════════════════════════════
# 교육용 더미 데이터 (Day3 ops_fixtures.json + Day4 ops_assets.json 을 합친 사본)
# ══════════════════════════════════════════════════════════════════
# 교육용 더미 데이터입니다. 다른 Day 폴더에 의존하지 않도록 이 파일 안에 그대로 옮겨 두었습니다.
# 도구는 결정적입니다. 같은 인자에는 항상 같은 결과를 돌려줍니다.

OPS_DATA: dict[str, Any] = {
    "_note": "교육용 더미 데이터입니다.",
    "deploys": {
        "payment-api": [
            {"version": "v2.14.1", "at": "2026-08-11T02:58:00+09:00", "by": "pay-be"},
            {"version": "v2.14.0", "at": "2026-08-08T14:20:00+09:00", "by": "pay-be"},
        ],
        "auth-service": [
            {"version": "v1.9.3", "at": "2026-08-05T11:00:00+09:00", "by": "iam"},
        ],
    },
    "logs": {
        "payment-api@2026-08-11T02:58:00+09:00": (
            "ERROR NullPointerException at PaymentValidator.java:88 (x1243)\n"
            "WARN circuit breaker open for ledger-api"
        ),
        "payment-api@default": "INFO health check ok",
        "auth-service@default": "INFO token issued x9821",
    },
    "metrics": {
        "payment-api": {"error_rate_5m": 0.124, "p99_latency_ms": 3400, "cpu": 0.42},
        "auth-service": {"error_rate_5m": 0.001, "p99_latency_ms": 180, "cpu": 0.30},
    },
    "runbooks": {
        "5xx": "1) 최근 30분 배포 이력 확인 2) 의존 서비스 상태 확인 3) DB 커넥션 풀 확인",
        "oom": "1) 힙 덤프 확보 2) 최근 메모리 설정 변경 확인 3) 롤백 검토",
        "cert": "1) 만료일 확인 2) 갱신 파이프라인 상태 확인 3) 수동 갱신 절차 수행",
    },
    "owners": {
        "payment-api": {"team": "pay-be", "oncall": "김온콜", "slack": "#pay-be-alert"},
        "auth-service": {"team": "iam", "oncall": "이보안", "slack": "#iam-alert"},
        "gateway": {"team": "platform", "oncall": "박게이트", "slack": "#platform-alert"},
    },
    "incidents": {
        "payment-api": [
            {"id": "INC-2026-0731", "title": "결제 승인 지연", "cause": "ledger-api 타임아웃", "at": "2026-07-31"},
            {"id": "INC-2026-0612", "title": "5xx 급증", "cause": "배포 회귀", "at": "2026-06-12"},
        ],
        "auth-service": [
            {"id": "INC-2026-0505", "title": "토큰 검증 실패", "cause": "인증서 만료", "at": "2026-05-05"},
        ],
    },
}


def _norm_service(service: Any) -> str:
    """서비스 이름을 소문자·하이픈 형태로 정리합니다. ('Payment API' -> 'payment-api')"""
    s = "" if service is None else str(service)
    return re.sub(r"[\s_]+", "-", s.strip().lower())


def _services_in(section: str) -> list[str]:
    """데이터 한 구역에서 조회 가능한 서비스 목록(정렬, 결정적)."""
    if section == "logs":
        return sorted({key.split("@", 1)[0] for key in OPS_DATA["logs"]})
    return sorted(OPS_DATA[section])


def _unknown_service(service: str, section: str) -> str:
    """서비스 이름이 비었거나 없을 때 모델이 다음 행동을 정할 수 있게 안내 문자열을 돌려줍니다."""
    available = ", ".join(_services_in(section))
    if not service:
        return f"서비스 이름이 비어 있습니다. 조회 가능한 서비스: {available}"
    return f"'{service}' 서비스의 데이터가 없습니다. 조회 가능한 서비스: {available}"


# ══════════════════════════════════════════════════════════════════
# 도메인별 도구 (도구는 예외를 던지지 않고, 설명 문자열을 돌려줍니다)
# ══════════════════════════════════════════════════════════════════
# 도구 docstring 에 '언제 쓰는지 / 언제 쓰지 말아야 하는지'를 함께 적어 둡니다.
# 각 도구는 정확히 한 Agent 에만 들어갑니다(도구 격리).

# ---------- log_agent 전용: "무엇이 찍혔는가" ----------

@tool
def get_logs(service: str, since: Optional[str] = None) -> str:
    """서비스 로그 원문을 조회합니다. since(ISO 8601 시각)를 주면 그 시점의 로그를 봅니다.

    에러 메시지·예외·스택트레이스처럼 '무엇이 찍혔는지' 확인할 때 사용합니다.
    시점을 모르면 search_error_logs 로 ERROR/WARN 이 찍힌 시점부터 찾으세요.
    지표(에러율·지연·CPU)나 대응 절차 조회에는 쓰지 않습니다.
    """
    name = _norm_service(service)
    logs = OPS_DATA["logs"]
    if not name or f"{name}@default" not in logs:
        return _unknown_service(name, "logs")
    since = (since or "").strip() or None
    if since and f"{name}@{since}" in logs:
        return f"[출처: get_logs] {name} {since} 시점 로그\n{logs[f'{name}@{since}']}"
    prefix = f"{name} {since} 시점 로그는 없어 평상시 로그를 보여 드립니다." if since else f"{name} 평상시 로그"
    return f"[출처: get_logs] {prefix}\n{logs[f'{name}@default']}"


@tool
def search_error_logs(service: str) -> str:
    """서비스의 모든 로그 구간에서 ERROR/WARN/FATAL 줄만 시점과 함께 추려 냅니다.

    '어떤 에러가 찍혔는지', '예외가 몇 건인지'를 시점을 모른 채 찾을 때 먼저 사용합니다.
    지표 수치나 대응 절차 조회에는 쓰지 않습니다.
    """
    name = _norm_service(service)
    if not name or f"{name}@default" not in OPS_DATA["logs"]:
        return _unknown_service(name, "logs")
    hits: list[str] = []
    for key in sorted(OPS_DATA["logs"]):
        svc, at = key.split("@", 1)
        if svc != name:
            continue
        for line in OPS_DATA["logs"][key].splitlines():
            if re.match(r"\s*(ERROR|WARN|FATAL)\b", line):
                hits.append(f"- [{at}] {line.strip()}")
    if not hits:
        return f"[출처: search_error_logs] {name}: ERROR/WARN 로그가 0건입니다 (모든 구간 확인)."
    return f"[출처: search_error_logs] {name}: ERROR/WARN {len(hits)}줄\n" + "\n".join(hits)


# ---------- metric_agent 전용: "얼마나, 언제" ----------

@tool
def get_metrics(service: str) -> str:
    """서비스의 현재 핵심 지표(5분 에러율, p99 지연시간, CPU 사용률)를 조회합니다.

    장애 규모·영향을 수치로 확인할 때 사용합니다.
    에러 메시지 원문(로그)이나 대응 절차 조회에는 쓰지 않습니다.
    """
    name = _norm_service(service)
    m = OPS_DATA["metrics"].get(name)
    if m is None:
        return _unknown_service(name, "metrics")
    return (
        f"[출처: get_metrics] {name} 현재 지표\n"
        f"- 5분 에러율: {m['error_rate_5m'] * 100:.1f}%\n"
        f"- p99 지연시간: {m['p99_latency_ms']}ms\n"
        f"- CPU 사용률: {m['cpu'] * 100:.0f}%"
    )


@tool
def get_deploy_history(service: str) -> str:
    """서비스의 최근 배포 이력(버전, 배포 시각, 배포 팀)을 최신순으로 조회합니다.

    지표가 변한 시점과 배포 시점을 맞춰 볼 때 사용합니다.
    배포 '절차'나 롤백 방법은 이 도구가 아니라 런북 담당이 답합니다.
    """
    name = _norm_service(service)
    deploys = OPS_DATA["deploys"].get(name)
    if not deploys:
        return _unknown_service(name, "deploys")
    ordered = sorted(deploys, key=lambda d: d["at"], reverse=True)
    lines = [f"[출처: get_deploy_history] {name} 배포 {len(ordered)}건 (최신순)"]
    lines += [f"- {d['version']} | {d['at']} | {d['by']}" for d in ordered]
    return "\n".join(lines)


# ---------- runbook_agent 전용: "어떻게, 누가" ----------

_RUNBOOK_ALIASES: dict[str, tuple[str, ...]] = {
    "5xx": ("5xx", "500", "502", "503", "504", "server error", "서버 에러", "서버 오류"),
    "oom": ("oom", "out of memory", "outofmemory", "memory", "메모리", "힙", "heap"),
    "cert": ("cert", "certificate", "ssl", "tls", "인증서", "x509"),
}


@tool
def search_runbook(symptom: str) -> str:
    """증상 키워드(5xx, oom, cert)로 장애 대응 런북(조치 절차)을 찾습니다.

    '무엇부터 조치해야 하는지'가 필요할 때 사용합니다. '500 에러', '메모리 부족',
    '인증서 만료' 같은 표현도 가장 가까운 키워드로 해석합니다.
    담당자 조회는 get_service_owner, 과거 장애 이력은 query_recent_incidents 를 쓰세요.
    """
    query = ("" if symptom is None else str(symptom)).strip().lower()
    runbooks = OPS_DATA["runbooks"]
    available = ", ".join(runbooks)
    if not query:
        return f"증상 키워드가 비어 있습니다. 쓸 수 있는 값: {available}"
    matched = next((k for k, aliases in _RUNBOOK_ALIASES.items() if any(a in query for a in aliases)), None)
    if matched is None:
        return f"'{query}' 에 해당하는 런북이 없습니다. 쓸 수 있는 값: {available}"
    return f"[출처: 런북 {matched}] {runbooks[matched]}"


@tool
def get_service_owner(service: str) -> str:
    """서비스의 담당팀, 온콜 담당자, 알림 Slack 채널을 조회합니다.

    '누구에게 연락/에스컬레이션하는지' 정할 때 사용합니다. 조치 절차는 search_runbook 을 쓰세요.
    """
    name = _norm_service(service)
    o = OPS_DATA["owners"].get(name)
    if o is None:
        return _unknown_service(name, "owners")
    return (
        f"[출처: get_service_owner] {name} 담당팀 {o['team']}, "
        f"온콜 {o['oncall']}, 알림 채널 {o['slack']}"
    )


@tool
def query_recent_incidents(service: str) -> str:
    """서비스의 과거 인시던트 이력(ID, 일자, 제목, 원인)을 최신순으로 조회합니다.

    비슷한 장애가 전에 있었는지, 그때 원인이 무엇이었는지 확인할 때 사용합니다.
    """
    name = _norm_service(service)
    known = sorted(set(OPS_DATA["owners"]) | set(OPS_DATA["incidents"]))
    if name not in known:
        return f"'{name}' 는 등록되지 않은 서비스입니다. 등록된 서비스: {', '.join(known)}"
    rows = sorted(OPS_DATA["incidents"].get(name, []), key=lambda r: r["at"], reverse=True)
    if not rows:
        return f"[출처: query_recent_incidents] {name}: 기록된 인시던트 0건"
    lines = [f"[출처: query_recent_incidents] {name}: {len(rows)}건 (최신순)"]
    lines += [f"- {r['at']} {r['id']} {r['title']} (원인: {r['cause']})" for r in rows]
    return "\n".join(lines)


# 이름 -> 도구 객체. AGENT_TOOLS 에 적힌 이름만 해당 Agent 에 꽂습니다(선언 = 실제 배선).
_TOOLS_BY_NAME = {
    t.name: t
    for t in (
        get_logs,
        search_error_logs,
        get_metrics,
        get_deploy_history,
        search_runbook,
        get_service_owner,
        query_recent_incidents,
    )
}


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# 이름이 곧 담당 범위의 선언입니다. 답이 이상하면 어느 프롬프트를 열어야 하는지 알려 주는 표지판.
AGENT_NAMES: list[str] = ["log_agent", "metric_agent", "runbook_agent"]

# 도메인 기준으로 도구를 나눕니다. 같은 도구가 두 Agent 에 들어가지 않습니다.
#   log_agent     : 무엇이 찍혔는가 (로그 원문, 에러 줄)
#   metric_agent  : 얼마나·언제 (수치 지표, 배포 시점)  ← README 라우팅 표에서 '배포'는 지표 쪽
#   runbook_agent : 어떻게·누가 (조치 절차, 담당자, 과거 사례)
AGENT_TOOLS: dict[str, list[str]] = {
    "log_agent": ["get_logs", "search_error_logs"],
    "metric_agent": ["get_metrics", "get_deploy_history"],
    "runbook_agent": ["search_runbook", "get_service_owner", "query_recent_incidents"],
}

# 워커별 시스템 프롬프트. 담당 범위를 좁게, 근거(수치·출처)를 반드시 남기도록 지시합니다.
# 근거를 남기라는 지시는 Supervisor 게이트(judge_output)와 짝을 이룹니다.
_COMMON_RULES = (
    "\n[공통 규칙]\n"
    "- 반드시 도구로 조회한 결과에 근거해서만 답한다. 도구 결과에 없는 내용은 추측하지 않는다.\n"
    "- 답에는 수치(건수·비율·시각·버전)와 출처(도구 이름 또는 런북 키)를 함께 적는다.\n"
    "- 근거 없이 '확실합니다', '틀림없습니다' 같은 단정 표현을 쓰지 않는다.\n"
    "- 담당 범위 밖 요청은 다루지 말고 '담당 범위 밖: <무엇>' 한 줄만 덧붙인다.\n"
    "- 3~5줄로 간결하게 보고한다."
)

AGENT_PROMPTS: dict[str, str] = {
    "log_agent": (
        "너는 인시던트 대응팀의 로그 분석 담당(log_agent)이다.\n"
        "로그 원문, 에러 메시지, 예외, 스택트레이스 확인만 담당한다.\n"
        "시점을 모르면 search_error_logs 로 ERROR/WARN 줄과 시점을 먼저 찾고, "
        "필요하면 get_logs(since=그 시점)로 원문을 본다." + _COMMON_RULES
    ),
    "metric_agent": (
        "너는 인시던트 대응팀의 지표 분석 담당(metric_agent)이다.\n"
        "에러율, p99 지연, CPU 같은 수치 지표와 배포 이력(버전·시각) 확인만 담당한다.\n"
        "지표 변화가 배포와 겹치는지 시각을 함께 비교해 보고한다." + _COMMON_RULES
    ),
    "runbook_agent": (
        "너는 인시던트 대응팀의 런북 담당(runbook_agent)이다.\n"
        "대응 절차(search_runbook), 담당팀·온콜(get_service_owner), "
        "과거 인시던트(query_recent_incidents) 조회만 담당한다.\n"
        "절차는 런북에 적힌 순서 그대로 전달한다." + _COMMON_RULES
    ),
}


# ── 라우팅 규칙 ────────────────────────────────────────────────────
# 키워드는 README 라우팅 표 + 동의어 + 영문 표기입니다.
# 정규식 하나에 한 '뜻'을 담고, 오탐이 나기 쉬운 단어는 앞뒤 문맥을 함께 겁니다.
#   - '로그인/로그아웃/블로그/카탈로그/백로그'는 로그가 아님
#   - '에러율/오류율/error rate'는 로그가 아니라 지표 (먼저 지표로 잡고 문장에서 지워 둠)
#   - '500ms'처럼 단위가 붙은 500은 HTTP 상태 코드가 아님

# 한글 조사가 영단어에 바로 붙으면("log를", "latency가") 파이썬 정규식의 \b 가 경계로 보지 않습니다.
# (한글도 \w 이기 때문) 그래서 영단어 경계는 '앞뒤가 영문자가 아님'으로 직접 겁니다.
def _en(words: str) -> str:
    """영단어 alternation 을 '앞뒤가 영문자가 아닌' 경계로 감쌉니다. 예: _en('logs?|logging')"""
    return rf"(?<![a-z])(?:{words})(?![a-z])"


# 비율형 지표 표현: 지표로 잡은 뒤 문장에서 지워 로그 키워드로 다시 걸리지 않게 합니다.
_RATE_RE = re.compile(
    r"(?:에러|오류|실패|장애|성공|타임아웃|에러\s*발생|오류\s*발생)\s*(?:율|률|비율|비중|%)"
    r"|(?<![a-z])(?:error|failure|success|timeout|5xx|4xx)s?\s*(?:rates?|ratio|percentage|%)"
)

_LOG_RE = re.compile(
    r"(?<![블탈얼롤필백])로그(?!인|아웃)"                    # 로그 (로그인·블로그·카탈로그·백로그 제외)
    r"|" + _en(r"logs?|logging|logged|logger|syslog|stderr|stdout|kibana|elk|loki|splunk") +
    r"|키바나|에러|오류|예외|익셉션|exception|" + _en(r"errors?|fail(?:ed|ures?|s)?|warn(?:ing)?s?|fatal") +
    r"|스택|stack\s*trace|stacktrace|traceback|트레이스|" + _en(r"traces?|tracing") + r"|추적"
    # HTTP 상태 코드: 500/502/503/504 (500ms·500건 처럼 단위가 붙으면 제외), 5xx, 4xx
    r"|(?<![\d.])5\d\d(?![\d.])(?!\s*(?:ms|밀리|초|s(?![a-z])|%|mb|gb|kb|건|개|명|원|번|회|대|줄|만|천))"
    r"|5xx|4xx|50x|" + _en(r"http\s*5\d\d|nullpointer\w*|npe|panic|segfault|crash(?:ed|es)?|core\s*dump") +
    r"|크래시|코어\s*덤프|경고\s*(?:로그|메시지)|찍힌|찍혔|찍혀|찍히"
)

_METRIC_RE = re.compile(
    r"지표|메트릭|수치|통계|" + _en(r"metrics?|stats?|statistics|measure(?:ment)?s?|kpi") +
    r"|지연|레이턴시|응답\s*(?:시간|속도)|느려|느림|느린|속도|" + _en(r"latency|response\s*time|slow(?:ness)?|lag") +
    r"|" + _en(r"p(?:50|75|90|95|99|999)") + r"|퍼센타일|percentile|백분위"
    r"|씨피유|메모리|램\s|힙\s*사용|디스크|" + _en(r"cpu|memory|mem|ram|heap\s*usage|disk|iops|gc") +
    r"|사용률|사용량|점유율|부하|" + _en(r"utili[sz]ation|usage|load|saturation") +
    r"|처리량|트래픽|요청\s*수|호출\s*수|" + _en(r"throughput|tps|qps|rps|traffic|requests?\s*per") +
    r"|가용성|성능|" + _en(r"availability|uptime|sl[oia]s?|performance") +
    r"|대시보드|그라파나|프로메테우스|" + _en(r"dashboards?|grafana|prometheus|datadog|cloudwatch")
)

# '배포'는 지표 담당이지만, 절차를 묻는 질문("배포 절차가 어떻게 되나요")은 런북 담당입니다.
_DEPLOY_RE = re.compile(
    r"배포|디플로이|릴리스|릴리즈|롤아웃|" + _en(r"deploy(?:s|ed|ment|ments)?|releases?|rollouts?")
)
# 절차를 묻는 표현
_PROCEDURE_RE = re.compile(
    r"절차|방법|매뉴얼|메뉴얼|런북|가이드|지침|플레이북|순서|단계|어떻게|"
    + _en(r"how|procedures?|runbooks?|playbooks?|guides?|manuals?|sop|steps?")
)
# 배포 '사실'(언제·무엇이·누가)을 묻는 표현 — 함께 있으면 절차 질문이어도 지표 담당이 필요합니다.
_DEPLOY_FACT_RE = re.compile(
    r"이력|내역|기록|히스토리|시각|시점|일시|언제|버전|최근|직후|이후|전후|마지막|"
    + _en(r"history|when|versions?|recent|last|latest")
)

_RUNBOOK_RE = re.compile(
    r"절차|대응|대처|조치|방법|런북|매뉴얼|메뉴얼|가이드|지침|플레이북|순서|단계별|처리\s*방안"
    r"|해결|복구|복원|완화|롤백|에스컬레이션|" + _en(r"escalat\w*|rollback") +
    r"|담당|책임자|오너|온콜|당번|연락|누구|누가|"
    + _en(r"owners?|on-?call|contacts?|who|whom|responsible") +
    r"|" + _en(r"runbooks?|playbooks?|manuals?|guides?|procedures?|sop|steps?|how\s+(?:to|do|can|should)")
    + r"|" + _en(r"mitigat\w*|remediat\w*|recover\w*|fix|resolv\w*|handle|respond|response\s*plan")
    + r"|과거\s*(?:장애|인시던트|사례)|인시던트\s*이력|장애\s*이력|이전\s*(?:장애|사례)|재발|포스트모템|회고"
    + r"|" + _en(r"postmortem|incident\s*history")
    + r"|(?:뭘|뭐|무엇을|뭣부터|무엇부터|뭐부터)\s*(?:해야|하면|할까)"
)
# '어떻게'는 다른 주제 없이 혼자 쓰였거나 행동 동사와 붙을 때만 런북입니다.
# ("지연이 어떻게 변했나요"는 지표 질문, "인증서 만료되면 어떻게 해?"는 런북 질문)
_HOW_RE = re.compile(r"어떻게|어케|어찌")
_HOW_ACTION_RE = re.compile(r"(?:어떻게|어케|어찌)\s*(?:해|하|대응|대처|조치|처리|복구|해결|고치|막|조치)")

# "지표만 알려주세요" → 그 담당만. "지표는 빼고/말고/필요 없고" → 그 담당 제외.
_ONLY_SUFFIX = re.compile(r"\s*만(?![료약큼족일원건명개번])")
_EXCLUDE_SUFFIX = re.compile(r"\s*(?:은|는|을|를|이|가)?\s*(?:빼고|제외|말고|필요\s*없|안\s*봐도)")

_MAX_QUESTION_CHARS = 4000  # 채널에 붙여 넣은 긴 로그 덤프도 앞부분만 보고 결정 (성능·결정성)


def _normalize_question(question: Any) -> str:
    """라우팅 전에 질문을 정리합니다. None·숫자·전각 문자도 예외 없이 처리합니다."""
    if question is None:
        return ""
    text = question if isinstance(question, str) else str(question)
    text = unicodedata.normalize("NFKC", text)[:_MAX_QUESTION_CHARS]
    return re.sub(r"\s+", " ", text).strip().lower()


def _spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    return [m.span() for m in pattern.finditer(text)]


def route_question(question: str) -> list[str]:
    """질문을 처리할 Agent 목록을 반환합니다. LLM을 호출하지 않습니다.

    반환 규칙:
      - 로그·에러·예외·스택·트레이스·5xx, 영문 log/exception/trace -> log_agent
      - 지표·수치·지연·레이턴시·CPU·메모리·오류율·배포, 영문 metric/latency -> metric_agent
      - 절차·대응·방법·런북·매뉴얼·담당·누구·어떻게, 영문 runbook -> runbook_agent
      - 여러 주제가 섞이면 해당하는 Agent를 모두 반환
      - 어디에도 해당하지 않으면 빈 목록 (Supervisor가 직접 답한다)

    **필요한 것만 반환하세요.** 전부 부르면 정확도는 오르지만 호출 비용이 폭증합니다.

    결정 순서:
      1) 정규화(NFKC, 소문자, 공백 정리, 길이 상한) — 어떤 입력에도 예외 없이 리스트 반환
      2) 비율형 표현(에러율·오류율·error rate)은 지표로 잡고 문장에서 지움 → 로그로 이중 배정 방지
      3) 카테고리별 키워드 매칭
      4) 문맥 보정: '배포 절차'는 런북만, '어떻게'는 단독/행동 동사일 때만 런북
      5) 범위 한정: 'X만'이면 X만, 'X는 빼고/말고'면 X 제외
      6) AGENT_NAMES 순서로 중복 없이 반환
    """
    try:
        text = _normalize_question(question)
        if not text:
            return []

        # 2) 비율형 표현 -> 지표. 지운 자리는 공백으로 두어 단어 경계를 유지합니다.
        rate_spans = _spans(_RATE_RE, text)
        scan = _RATE_RE.sub(lambda m: " " * len(m.group(0)), text)  # 길이 유지(위치 보존)

        # 3) 카테고리별 매칭 (담당별로 어떤 위치에서 걸렸는지 기록)
        log_spans = _spans(_LOG_RE, scan)
        metric_spans = rate_spans + _spans(_METRIC_RE, scan)
        deploy_spans = _spans(_DEPLOY_RE, scan)
        runbook_spans = _spans(_RUNBOOK_RE, scan)
        how_spans = _spans(_HOW_RE, scan)

        # 4-a) '어떻게'
        if how_spans and not runbook_spans:
            if _HOW_ACTION_RE.search(scan) or not (log_spans or metric_spans or deploy_spans):
                runbook_spans = how_spans

        # 4-b) '배포': 절차를 묻는 질문이면 런북만, 배포 사실(이력·시각·버전)을 물으면 지표도.
        if deploy_spans and not (_PROCEDURE_RE.search(scan) and not _DEPLOY_FACT_RE.search(scan)):
            metric_spans = metric_spans + deploy_spans

        spans_by_agent = {
            "log_agent": log_spans,
            "metric_agent": metric_spans,
            "runbook_agent": runbook_spans,
        }
        hits = {agent for agent, sp in spans_by_agent.items() if sp}

        # 5-a) 'X만': '만'이 붙은 담당만 남깁니다. ("지표만 알려주세요" -> metric_agent)
        only = {
            agent for agent, sp in spans_by_agent.items()
            if any(_ONLY_SUFFIX.match(scan, end) for _, end in sp)
        }
        if only:
            hits &= only
        # 5-b) 'X는 빼고/말고': 그 담당의 모든 언급이 제외 표현과 붙었을 때만 뺍니다.
        excluded = {
            agent for agent, sp in spans_by_agent.items()
            if sp and all(_EXCLUDE_SUFFIX.match(scan, end) for _, end in sp)
        }
        if excluded and hits - excluded:  # 전부 지워지는 경우는 원래 결과 유지(안전)
            hits -= excluded

        # 6) 고정 순서 + 중복 없음
        return [name for name in AGENT_NAMES if name in hits]
    except Exception:  # noqa: BLE001 - 라우터가 예외를 던지면 뒤의 Agent 셋은 실행조차 못 합니다.
        return []


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

_MIN_USEFUL_CHARS = 20

# 근거 없이 단정하는 표현 (한국어 + 영문)
_ASSERTIVE_RE = re.compile(
    r"확실|틀림\s*없|틀림없|분명(?:히|합니다|하다)|명백|자명|단언|장담|확신|의심의\s*여지|"
    r"무조건|100\s*%\s*(?:확실|맞)|당연히|절대(?:로)?\s|그것뿐|그것\s*밖에|이\s*외에는\s*없|"
    r"definitely|certainly|undoubtedly|without\s+(?:a\s+)?doubt|guaranteed|surely|"
    r"absolutely|obviously|clearly\s+(?:the|caused|due)",
    re.IGNORECASE,
)

# 근거로 인정하는 것: 수치 또는 출처 표기
_EVIDENCE_RE = re.compile(
    r"\d"                                                   # 수치(건수·비율·시각·버전·코드)
    r"|출처|근거\s*[:：]|\[런북|런북\s*[:：]|according\s+to|source\s*[:：]|https?://"
    r"|get_logs|search_error_logs|get_metrics|get_deploy_history"
    r"|search_runbook|get_service_owner|query_recent_incidents"
    r"|\.(?:java|py|go|js|ts|kt|md|log)\b|#[\w-]+-alert|INC-",
    re.IGNORECASE,
)


def judge_output(agent_name: str, output: str) -> dict:
    """워커 산출물을 내보내기 전에 걸러냅니다. LLM을 호출하지 않습니다.

    워커가 만든 것을 그대로 사용자에게 주면 근거 없는 단정과 빈 답변이 섞입니다.
    Supervisor는 게이트 역할을 해야 합니다.

    판정 규칙:
      - 내용이 비었거나 20자 미만          -> {"keep": False, "reason": "저가치"}
      - "확실합니다" "틀림없습니다" 같은
        근거 없는 단정이 있는데 수치·출처가 없음 -> {"keep": False, "reason": "근거 없는 단정"}
      - 그 외                              -> {"keep": True, "reason": ...}

    Returns:
        {"keep": bool, "reason": str} 형태의 dict
        (추적용으로 "agent", "detail" 키를 함께 남깁니다. 다음 날 게이트가 과하게 잠긴 건지
         워커가 부실한 건지 구분하려면 사유가 구체적이어야 합니다.)
    """
    agent = str(agent_name or "unknown")
    text = get_text(output) if not isinstance(output, str) else output
    text = (text or "").strip()
    length = len(text)

    # 규칙 1) 빈 값 / 20자 미만
    if length < _MIN_USEFUL_CHARS:
        detail = "내용이 비어 있음" if length == 0 else f"{length}자 (최소 {_MIN_USEFUL_CHARS}자 미만)"
        return {"keep": False, "reason": "저가치", "agent": agent, "detail": detail}

    # 규칙 2) 단정 표현 + 근거(수치·출처) 없음
    assertive = _ASSERTIVE_RE.findall(text)
    if assertive and not _EVIDENCE_RE.search(text):
        words = ", ".join(dict.fromkeys(a.strip() for a in assertive))
        return {
            "keep": False,
            "reason": "근거 없는 단정",
            "agent": agent,
            "detail": f"단정 표현({words})이 있으나 수치·출처가 없음",
        }

    # 규칙 3) 통과
    detail = "단정 표현에 수치·출처 근거가 있음" if assertive else f"{length}자, 단정 표현 없음"
    return {"keep": True, "reason": "통과", "agent": agent, "detail": detail}


# ══════════════════════════════════════════════════════════════════
# Supervisor 조립 (langgraph_supervisor)
# ══════════════════════════════════════════════════════════════════

SUPERVISOR_PROMPT = (
    "너는 인시던트 대응팀의 Supervisor다. 로그(log_agent), 지표(metric_agent), "
    "런북(runbook_agent) 담당의 보고를 받아 사용자에게 하나의 답으로 종합한다.\n"
    "[규칙]\n"
    "- 워커 보고에 있는 사실만 쓴다. 보고에 없는 수치나 원인을 지어내지 않는다.\n"
    "- 게이트에서 억제된 보고는 내용을 쓰지 말고 '어느 담당의 보고가 왜 억제됐는지'만 밝힌다.\n"
    "- 보고 사이에 시각이 겹치면(배포 시각과 에러 발생 시각 등) 연관 가능성으로 제시하되 단정하지 않는다.\n"
    "- 결론 → 근거(담당별) → 다음 조치 순서로 짧게 정리한다.\n"
    "- 워커 보고가 하나도 없으면(인시던트 대응과 무관한 질문) 담당 범위가 "
    "로그·지표·런북 조회임을 한두 문장으로 안내한다."
)

_GATE_MARK = "sds_gate"  # 메시지 response_metadata 에 남기는 게이트 판정 표식
_HANDOFF_PREFIX = "transfer_to_"


def _last_human_index(messages: list) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            return i
    return -1


def _is_worker_report(message: Any) -> bool:
    """워커가 낸 최종 보고(AIMessage)인지. handoff-back 메시지와 도구 호출 메시지는 제외."""
    from langgraph_supervisor.handoff import METADATA_KEY_IS_HANDOFF_BACK

    return (
        getattr(message, "type", None) == "ai"
        and getattr(message, "name", None) in AGENT_NAMES
        and not getattr(message, "tool_calls", None)
        and not (getattr(message, "response_metadata", None) or {}).get(METADATA_KEY_IS_HANDOFF_BACK)
    )


def _supervisor_gate(state: dict) -> dict:
    """Supervisor 의 pre_model_hook. 게이트는 여기 한 곳에만 있습니다.

    1) route_question 으로 이번 질문의 targets 를 계산해 state 에 남깁니다.
    2) 직전에 돌아온 워커 보고를 judge_output 으로 판정합니다.
       억제된 보고는 같은 id 의 메시지로 교체(add_messages 가 id 로 덮어씀)해,
       Supervisor 모델도 최종 출력도 원문 대신 '억제 사유'를 보게 합니다.
    """
    messages = list(state.get("messages") or [])
    start = _last_human_index(messages)
    question = get_text(messages[start]) if start >= 0 else ""

    gate_log = list(state.get("gate_log") or [])
    replaced: list = []
    for m in messages[start + 1 :]:
        meta = getattr(m, "response_metadata", None) or {}
        if not _is_worker_report(m) or _GATE_MARK in meta:
            continue  # 워커 보고가 아니거나 이미 판정한 메시지
        verdict = judge_output(m.name, get_text(m))
        gate_log.append(verdict)
        new_meta = {**meta, _GATE_MARK: verdict}
        if verdict["keep"]:
            replaced.append(m.model_copy(update={"response_metadata": new_meta}))
        else:
            replaced.append(
                AIMessage(
                    id=m.id,
                    name=m.name,
                    content=(
                        f"[게이트 억제] {m.name} 산출물을 내보내지 않았습니다. "
                        f"사유: {verdict['reason']} ({verdict['detail']})"
                    ),
                    response_metadata=new_meta,
                )
            )

    return {
        "messages": replaced,  # 같은 id 는 교체, 나머지는 그대로 (add_messages)
        "targets": route_question(question),
        "gate_log": gate_log,
    }


class _RuleRoutedSupervisorModel(BaseChatModel):
    """langgraph_supervisor 에 넣는 Supervisor '모델'입니다.

    BaseChatModel 인터페이스를 따르지만, 배분은 LLM 이 아니라 route_question 이 합니다.
      - 아직 부르지 않은 target 이 있으면: transfer_to_<agent> 도구 호출을 결정적으로 생성
        (한 번에 하나씩, AGENT_NAMES 순서로. 병렬 Send handoff 는 이 버전에서 한 갈래만 살아남아 쓰지 않음)
      - 모든 target 이 보고를 마쳤으면: synthesizer(실제 LLM)로 최종 답 1회 생성
      - 전부 게이트에서 억제됐으면: LLM 없이 억제 사유를 그대로 알림(비용 0)

    배분까지 모델에게 맡기면 오답이 났을 때 워커가 틀렸는지 배분이 틀렸는지 가릴 수 없습니다.
    """

    synthesizer: Any = None
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "rule-routed-supervisor"

    def bind_tools(self, tools, *, parallel_tool_calls: bool | None = None, **kwargs: Any):
        # handoff 도구 목록은 langgraph_supervisor 가 ToolNode 로 따로 실행합니다.
        # 여기서는 이름 규칙(transfer_to_<agent>)만 쓰므로 자기 자신을 돌려줍니다.
        return self

    # ---- 내부 도우미 ----
    @staticmethod
    def _plan(messages: list) -> tuple[str, list[str], list[str]]:
        start = _last_human_index(messages)
        question = get_text(messages[start]) if start >= 0 else ""
        targets = route_question(question)
        # 이미 다녀간 담당: 워커가 남긴 메시지(보고·handoff back)의 name, 또는 handoff 도구 호출.
        # (병렬 Send handoff 였다면 transfer_to_* 메시지가 부모 상태에 남지 않으므로 둘 다 봅니다.)
        called: set[str] = set()
        for m in messages[start + 1 :]:
            if getattr(m, "type", None) == "ai" and getattr(m, "name", None) in AGENT_NAMES:
                called.add(m.name)
            for tc in getattr(m, "tool_calls", None) or []:
                name = tc["name"] if isinstance(tc, dict) else tc.name
                if name.startswith(_HANDOFF_PREFIX):
                    called.add(name[len(_HANDOFF_PREFIX) :])
        pending = [t for t in targets if t not in called]
        return question, targets, pending

    def _answer(self, messages: list) -> AIMessage:
        start = _last_human_index(messages)
        question = get_text(messages[start]) if start >= 0 else ""
        reports = [m for m in messages[start + 1 :] if _is_worker_report(m)]
        kept = [m for m in reports if (m.response_metadata.get(_GATE_MARK) or {}).get("keep", True)]

        # 모든 보고가 억제됐으면 LLM 을 부를 이유가 없습니다. 사유를 그대로 전달합니다.
        if reports and not kept:
            lines = [get_text(m) for m in reports]
            return AIMessage(
                content="담당 Agent 보고가 모두 게이트에서 억제되어 확정된 답을 드릴 수 없습니다.\n"
                + "\n".join(f"- {line}" for line in lines)
            )

        system = next(
            (get_text(m) for m in messages if getattr(m, "type", None) == "system"),
            SUPERVISOR_PROMPT,
        )
        if reports:
            body = "\n\n".join(f"[{m.name} 보고]\n{get_text(m)}" for m in reports)
            user = f"사용자 질문: {question}\n\n{body}\n\n위 보고만 근거로 종합 답변을 작성하라."
        else:
            user = f"사용자 질문: {question}\n\n(라우팅 규칙상 담당 Agent 없음. 직접 답하라.)"
        # 도구 호출/도구 결과 블록을 빼고 텍스트만 넘깁니다.
        # (Bedrock 은 toolConfig 없이 toolUse 블록이 섞인 대화를 거부합니다.)
        reply = self.synthesizer.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return AIMessage(content=get_text(reply) or "(빈 응답)")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        _, targets, pending = self._plan(list(messages))
        if pending:
            message = AIMessage(
                content=f"라우팅 규칙: {targets} → 이번 호출 {pending[0]} (남은 담당 {pending[1:]})",
                tool_calls=[
                    {
                        "name": f"{_HANDOFF_PREFIX}{name}",
                        "args": {},
                        "id": f"call_route_{name}_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    }
                    for name in pending[:1]  # 한 번에 한 담당씩 (결정적 순서)
                ],
            )
        else:
            message = self._answer(list(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])


# 모듈이 sys.modules 에 등록되지 않은 채 로드돼도(채점기 방식) 필드 타입을 풀 수 있게 명시합니다.
_RuleRoutedSupervisorModel.model_rebuild(_types_namespace={"Any": Any})


def _make_router_model(synthesizer: Any) -> _RuleRoutedSupervisorModel:
    """규칙 라우터 + 종합용 LLM 을 묶은 Supervisor 모델을 만듭니다."""
    return _RuleRoutedSupervisorModel(synthesizer=synthesizer)


def _state_schema():
    """Supervisor 그래프 State. messages 외에 라우팅 결과와 게이트 기록을 남깁니다.

    gate_log 는 reducer 없이 덮어씁니다. supervisor 서브그래프가 전체 상태를 돌려주므로
    operator.add 를 쓰면 같은 기록이 매 턴 중복으로 쌓입니다.
    """
    from langgraph.prebuilt.chat_agent_executor import AgentState

    class IncidentState(AgentState):
        targets: list[str]
        gate_log: list[dict]

    return IncidentState


def build_supervisor(llm=None):
    """Supervisor 그래프를 만들어 컴파일해 반환합니다.

    반드시 지킬 것:
      - AGENT_NAMES의 이름이 모두 그래프 노드로 존재해야 합니다.
      - supervisor 노드에서 각 Agent로 가는 경로가 있어야 합니다.
      - 종료 경로가 있어야 합니다.

    langgraph_supervisor를 써도 되고 StateGraph로 직접 만들어도 됩니다.

    여기서는 langgraph_supervisor.create_supervisor 로 조립합니다.
      - 워커: langchain create_agent 로 만든 전문 Agent 3개 (AGENT_TOOLS 에 적힌 도구만 장착)
      - supervisor 모델: 규칙 라우터(route_question) + 종합용 LLM
      - pre_model_hook: 게이트(judge_output) — 워커 산출물은 반드시 여기를 거칩니다
      - 경로: supervisor → 각 워커(handoff) → supervisor(handoff back) → END
    """
    llm = llm or _default_llm()

    from langchain.agents import create_agent
    from langgraph_supervisor import create_supervisor

    workers = [
        create_agent(
            llm,
            [_TOOLS_BY_NAME[t] for t in AGENT_TOOLS[name]],
            system_prompt=AGENT_PROMPTS[name],
            name=name,
        )
        for name in AGENT_NAMES
    ]

    workflow = create_supervisor(
        workers,
        model=_make_router_model(llm),
        prompt=SUPERVISOR_PROMPT,
        pre_model_hook=_supervisor_gate,
        state_schema=_state_schema(),
        output_mode="last_message",  # 워커 내부 도구 호출 기록은 빼고 최종 보고만 올림 (토큰 절약)
        parallel_tool_calls=False,  # 한 번에 한 담당씩 handoff (병렬 Send 는 한 갈래만 반영되는 문제)
        supervisor_name="supervisor",
    )
    return workflow.compile()


# ══════════════════════════════════════════════════════════════════
# 직접 실행해 보기 (실제 Bedrock 호출)
#   python day06/practice/starter/submission.py
# ══════════════════════════════════════════════════════════════════

def run(question: str, app=None) -> dict:
    """질문 하나를 실행하며 노드별 진행 상황과 게이트 판정을 출력합니다."""
    from langchain_core.messages import HumanMessage

    app = app or build_supervisor()
    print(f"\n질문: {question}")
    print(f"  라우팅(규칙): {route_question(question)}")
    final: dict = {}
    for event in app.stream(
        {"messages": [HumanMessage(content=question)], "targets": [], "gate_log": []},
        stream_mode="updates",
        config={"recursion_limit": 30},
    ):
        for node, update in event.items():
            for m in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                for tc in getattr(m, "tool_calls", None) or []:
                    print(f"  [{node}] 호출: {tc['name']}")
                if not getattr(m, "tool_calls", None) and getattr(m, "type", "") == "ai":
                    print(f"  [{getattr(m, 'name', None) or node}] {get_text(m)[:200]}")
            if isinstance(update, dict) and update.get("gate_log"):
                final["gate_log"] = update["gate_log"]
    for v in final.get("gate_log", []):
        print(f"  [게이트] {v['agent']}: keep={v['keep']} 사유={v['reason']} ({v['detail']})")
    return final


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    app = build_supervisor()
    for q in (
        "payment-api 로그에 어떤 에러가 찍혔나요?",
        "payment-api 오류율 지표를 알려주세요.",
        "인증서 만료 대응 절차가 어떻게 되나요?",
        "payment-api 로그의 에러와 현재 지표를 함께 봐 주세요.",
        "오늘 점심 메뉴 추천해줘",
    ):
        run(q, app)
