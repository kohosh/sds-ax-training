"""Day 7 — 계측과 누적 평가.

6일간 쌓은 시스템을 측정합니다.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. run_eval과 compare_eval은 LLM을 호출하지 않습니다.
     '답이 맞았는가'의 판정은 호출자가 넘겨 준 함수가 합니다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

# LangChain이 설치되어 있으면 콜백 핸들러를 상속해, TraceCollector를 그대로
# `config={"callbacks": [collector]}` 에 꽂을 수 있게 합니다.
# 설치되어 있지 않아도(채점 환경 등) 제출 파일이 불러와지도록 object로 대신합니다.
# (모듈 최상단에서는 '도구'만 가져오고 모델은 만들지 않습니다.)
try:  # pragma: no cover - 설치 여부에 따라 갈리는 분기
    from langchain_core.callbacks import BaseCallbackHandler as _CallbackBase
except Exception:  # noqa: BLE001
    _CallbackBase = object  # type: ignore[assignment,misc]

# 판정 결과에 쓰는 상태값입니다. 이 셋만 씁니다.
PASS, FAIL, ERROR = "pass", "fail", "error"

# TraceCollector가 받는 이벤트 종류입니다. 이 넷 외의 값은 오타로 보고 거절합니다.
# trace는 '결정적 구조체'여야 grep·집계·채점이 확정적으로 됩니다.
# "tool_end"를 "tools_end"로 잘못 적은 이벤트가 섞이면 개수 검사가 조용히 틀어집니다.
TRACE_EVENTS = ("llm_start", "llm_end", "tool_start", "tool_end")

# failures.detail 에 답변을 미리보기로 남길 때의 최대 길이입니다.
_PREVIEW_LEN = 120


def _preview(value: Any, limit: int = _PREVIEW_LEN) -> str:
    """답변·입력값을 사람이 읽을 수 있는 짧은 문자열로 줄입니다."""
    text = value if isinstance(value, str) else repr(value)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

class TraceCollector(_CallbackBase):
    """실행 중 일어난 일을 기록합니다.

    LangChain 콜백 핸들러로도 쓸 수 있게 만들되, 여기서는 최소 인터페이스만 요구합니다.
    trace는 결정적 구조체라서, LLM이 무슨 말을 했든 이벤트 개수와 필드는 확정 검사됩니다.

    이벤트 한 건의 모양 (한 줄 = 한 dict):
        {"seq": 1, "event": "tool_end", "name": "get_logs", "ts": 1731830400.12,
         "latency_s": 0.01, ...호출자가 넘긴 부가 필드}

    - seq   : 1부터 오르는 순번. 시계가 아니라 '기록된 순서'라 같은 실행은 항상 같은 값.
    - ts    : 기록 시각(유닉스 초). 다른 로그와 시간축을 맞출 때 씁니다.
    - latency_s : 호출자가 주면 그 값을, 안 주면 짝이 되는 *_start 이후 경과 시간을
                  자동으로 채웁니다(*_end 이벤트에 한함).

    사용법
        t = TraceCollector()                       # 메모리에만 기록
        t = TraceCollector("trace.jsonl")          # 기록할 때마다 파일에도 한 줄씩 append
        t.record("tool_start", "get_logs")
        t.record("tool_end", "get_logs")           # latency_s 자동 계산
        t.dump("trace.jsonl")                      # 전체를 JSON Lines로 저장
    """

    # LangChain 콜백 매니저가 '예외를 다시 던질지' 등을 확인하는 속성입니다.
    # 계측 코드의 오류가 본 실행을 멈추면 안 되므로 False로 둡니다.
    raise_error = False

    def __init__(self, live_path: str | Path | None = None) -> None:
        # 기록을 담을 목록. 순서가 곧 seq 입니다.
        self._events: list[dict] = []
        # 아직 끝나지 않은 *_start 들의 시작 시각.
        #   key = run_id(LangChain이 넘겨 줄 때) 또는 (종류, 이름)
        #   value = 시작 시각 스택 (같은 도구가 중첩 호출돼도 짝이 맞도록 스택으로 둡니다)
        self._open: dict[Any, list[float]] = {}
        # LangChain 콜백용: run_id → *_start 때의 모델·도구 이름
        self._run_names: dict[str, str | None] = {}
        # live_path를 주면 record 때마다 파일에 바로 한 줄씩 씁니다.
        # 실행이 중간에 죽어도 그때까지의 trace가 파일에 남게 하려는 장치입니다.
        self._live_path = Path(live_path) if live_path is not None else None
        if self._live_path is not None:
            self._live_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 기록 ────────────────────────────────────────────────────────

    def record(self, event: str, name: str, **fields: Any) -> None:
        """이벤트 하나를 기록합니다.

        event: "llm_start" | "llm_end" | "tool_start" | "tool_end" 중 하나
        name : 모델 이름 또는 도구 이름
        fields: latency_s, tokens 등 부가 정보
        """
        if event not in TRACE_EVENTS:
            raise ValueError(f"알 수 없는 event 입니다: {event!r}. 허용값: {', '.join(TRACE_EVENTS)}")

        now = time.perf_counter()
        kind, phase = event.split("_", 1)  # ("llm"|"tool", "start"|"end")
        # 짝 맞추기 키: run_id가 있으면 그것이 가장 정확하고, 없으면 (종류, 이름)으로 맞춥니다.
        run_id = fields.get("run_id")
        key = ("run", str(run_id)) if run_id is not None else (kind, str(name))

        if phase == "start":
            self._open.setdefault(key, []).append(now)
        else:
            stack = self._open.get(key)
            started = stack.pop() if stack else None
            if stack is not None and not stack:
                del self._open[key]
            # 호출자가 지연 시간을 직접 주지 않았을 때만 자동으로 채웁니다.
            if "latency_s" not in fields and started is not None:
                fields["latency_s"] = round(now - started, 4)

        # 핵심 필드(seq/event/name/ts)가 부가 필드에 덮이지 않도록 마지막에 씁니다.
        entry: dict[str, Any] = dict(fields)
        entry.update(
            seq=len(self._events) + 1,
            event=event,
            name=str(name),
            ts=round(time.time(), 3),
        )
        # 보기 좋게 핵심 필드를 앞으로 정렬합니다. (dict는 삽입 순서를 유지)
        ordered = {k: entry[k] for k in ("seq", "event", "name", "ts")}
        ordered.update((k, v) for k, v in entry.items() if k not in ordered)
        self._events.append(ordered)

        if self._live_path is not None:
            with self._live_path.open("a", encoding="utf-8") as f:
                f.write(self._to_line(ordered) + "\n")

    @property
    def events(self) -> list[dict]:
        """기록된 이벤트 목록을 반환합니다."""
        # 복사본을 돌려줍니다. 바깥에서 목록이나 dict를 고쳐도 원본 trace는 그대로 남습니다.
        return [dict(e) for e in self._events]

    def dump(self, path: str | Path) -> None:
        """이벤트를 JSON Lines 파일로 씁니다. 한 줄에 한 이벤트."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in self._events:
                f.write(self._to_line(e) + "\n")

    @staticmethod
    def _to_line(event: dict) -> str:
        # ensure_ascii=False: 한국어를 \uXXXX 로 깨지 않고 그대로 남겨 grep이 되게 합니다.
        # default=str: UUID·datetime 같은 값이 섞여도 dump가 죽지 않게 문자열로 바꿉니다.
        return json.dumps(event, ensure_ascii=False, default=str)

    # ── 요약 (어느 구간이 느린가) ──────────────────────────────────

    def summary(self) -> dict:
        """이벤트 종류별 개수와 이름별 지연 합계를 돌려줍니다.

        "느리다"는 체감을 "get_logs가 3번 불려 합계 2.1초"처럼 구간 단위로 바꿔 줍니다.
        """
        counts = {e: 0 for e in TRACE_EVENTS}
        latency: dict[str, dict[str, float]] = {}
        for e in self._events:
            counts[e["event"]] += 1
            if e["event"].endswith("_end") and isinstance(e.get("latency_s"), (int, float)):
                slot = latency.setdefault(e["name"], {"calls": 0, "total_s": 0.0})
                slot["calls"] += 1
                slot["total_s"] = round(slot["total_s"] + float(e["latency_s"]), 4)
        return {"counts": counts, "latency_by_name": latency}

    # ── LangChain 콜백 훅 ──────────────────────────────────────────
    # BaseCallbackHandler 의 메서드 이름을 그대로 따릅니다. LangChain이 없으면 그냥 안 불릴 뿐입니다.
    # 콜백 안에서 예외가 나도 본 실행을 막지 않도록 _safe 로 감쌉니다.

    def _safe(self, event: str, name: str | None, **fields: Any) -> None:
        try:
            # *_end 콜백은 이름을 알려 주지 않는 경우가 많아, *_start 때 run_id로 기억해 둔
            # 이름을 다시 씁니다. (llm_start "claude-…" ↔ llm_end "claude-…" 로 짝이 보이게)
            names = self._run_names
            run_id = fields.get("run_id")
            if run_id is not None:
                fields["run_id"] = str(run_id)  # UUID 객체 대신 문자열로 남겨 events·파일이 같은 모양이 되게
            if event.endswith("_start") and run_id is not None:
                names[str(run_id)] = name
            elif run_id is not None:
                name = names.pop(str(run_id), None) or name
            self.record(event, name or event.split("_")[0], **fields)
        except Exception:  # noqa: BLE001 - 계측 실패가 업무 실행을 멈추면 안 됩니다
            pass

    @staticmethod
    def _model_name(serialized: Any, kwargs: dict) -> str:
        params = kwargs.get("invocation_params") or {}
        serialized = serialized or {}
        return str(
            params.get("model") or params.get("model_id") or params.get("model_name")
            or (serialized.get("kwargs") or {}).get("model")
            or serialized.get("name") or "llm"
        )

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._safe("llm_start", self._model_name(serialized, kwargs), run_id=run_id)

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._safe("llm_start", self._model_name(serialized, kwargs), run_id=run_id)

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        tokens = None
        try:
            msg = getattr(response.generations[0][0], "message", None)
            tokens = getattr(msg, "usage_metadata", None) or None
        except Exception:  # noqa: BLE001
            pass
        fields = {"run_id": run_id}
        if tokens:
            fields["tokens"] = dict(tokens)
        self._safe("llm_end", None, **fields)

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._safe("tool_start", name, run_id=run_id, input=_preview(input_str, 200))

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        content = getattr(output, "content", output)
        name = getattr(output, "name", None) or kwargs.get("name")
        self._safe("tool_end", name, run_id=run_id, output=_preview(content, 200))


# ── run_eval 보조 함수 ───────────────────────────────────────────

class _MalformedItem(Exception):
    """평가셋 문항 자체가 잘못된 경우(시스템 예외와 구분하기 위한 표식)."""


def _is_nonempty(answer: Any) -> bool:
    """judge가 없을 때의 기본 판정: '답이 비어 있지 않은가'.

    - None → 빈 답
    - 문자열 → 공백만 있으면 빈 답
    - {"answer": ...} 형태(Day2 RAG 체인 반환값) → answer 필드로 판정
    - 리스트·dict 등 컨테이너 → 원소가 없으면 빈 답
    - 그 밖의 값(숫자, True 등) → 값이 있으므로 비어 있지 않음
    """
    if answer is None:
        return False
    if isinstance(answer, str):
        return bool(answer.strip())
    if isinstance(answer, dict) and "answer" in answer:
        return _is_nonempty(answer["answer"])
    if isinstance(answer, (list, tuple, set, dict)):
        return len(answer) > 0
    return True


def run_eval(answer_fn: Callable[[str], Any], eval_set: list[dict],
             judge: Callable[[dict, Any], bool] | None = None) -> dict:
    """평가셋을 전부 돌려 유형별 통과율을 냅니다.

    Args:
        answer_fn: 질문 문자열을 받아 답을 돌려주는 함수. 예외가 날 수 있습니다.
        eval_set: [{"id", "question", "type", ...}] 목록
        judge: (문항, 답) -> 통과 여부. None이면 '답이 비어 있지 않으면 통과'로 봅니다.

    Returns:
        {
          "total": 전체 문항 수,
          "passed": 통과 수,
          "by_type": {유형: {"total": n, "passed": m}},
          "failures": [{"id", "type", "status", "detail"}],
        }

    반드시 지킬 것:
      - answer_fn이 예외를 던져도 전체 실행이 멈추면 안 됩니다.
        그 문항은 status를 ERROR로 기록하고 계속 진행합니다.

    추가 반환 키:
      - "results": [{"id", "type", "status", "latency_s"}] — 모든 문항의 상태를 평가셋 순서대로.
        failures만으로는 '통과한 문항 id'를 알 수 없어서, compare_eval 이 문항 단위로
        전후를 맞대기 위해 씁니다. (규정 스키마의 네 키는 그대로 유지합니다)
    """
    items = list(eval_set or [])
    by_type: dict[str, dict[str, int]] = {}
    failures: list[dict] = []
    results: list[dict] = []
    passed = 0

    for idx, item in enumerate(items):
        # 문항이 dict가 아니거나 id가 빠져도 집계에서 사라지면 안 됩니다.
        # id가 없으면 위치 기반 임시 id를 붙여 failures에 반드시 남깁니다.
        if not isinstance(item, dict):
            qid, qtype, question = f"#{idx}", "unknown", None
        else:
            qid = item.get("id") or f"#{idx}"
            qtype = item.get("type") or "unknown"
            question = item.get("question", "")

        slot = by_type.setdefault(qtype, {"total": 0, "passed": 0})
        slot["total"] += 1

        status, detail = PASS, ""
        started = time.perf_counter()
        try:
            if question is None:
                # 평가셋 자체의 결함. answer_fn을 부르지 않고 error로 남깁니다.
                raise _MalformedItem(f"문항 형식 오류 — dict가 아닙니다 ({type(item).__name__})")
            # ① 답 생성: 여기서 터지면 '시스템'의 결함입니다.
            answer = answer_fn(question)
        except _MalformedItem as exc:
            status, detail = ERROR, str(exc)
        except Exception as exc:  # noqa: BLE001 - 한 문항의 예외가 전체를 멈추면 안 됩니다
            status, detail = ERROR, f"answer_fn 예외 — {type(exc).__name__}: {exc}"
        else:
            # ② 판정: 여기서 터지면 '채점 기준(judge)'의 결함이라 detail을 따로 적습니다.
            #    틀린 답(fail)과 터진 코드(error)는 고칠 사람도 자리도 다르므로 나눠 기록합니다.
            try:
                ok = bool(judge(item, answer)) if judge is not None else _is_nonempty(answer)
            except Exception as exc:  # noqa: BLE001
                status, detail = ERROR, f"judge 예외 — {type(exc).__name__}: {exc}"
            else:
                if not ok:
                    reason = "judge 불통과" if judge is not None else "빈 답변"
                    status, detail = FAIL, f"{reason}: answer={_preview(answer)}"
        latency = round(time.perf_counter() - started, 4)

        if status == PASS:
            passed += 1
            slot["passed"] += 1
        else:
            failures.append({"id": qid, "type": qtype, "status": status, "detail": detail})
        results.append({"id": qid, "type": qtype, "status": status, "latency_s": latency})

    return {
        "total": len(items),
        "passed": passed,
        "by_type": by_type,
        "failures": failures,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

def _status_map(run: dict) -> dict[str, str]:
    """run_eval 결과에서 {문항 id: 상태} 를 평가셋 순서대로 뽑습니다.

    - "results"가 있으면(이 파일의 run_eval) 모든 문항의 상태를 그대로 씁니다.
    - 없으면(규정 스키마만 지킨 다른 run_eval) failures만 알 수 있으므로,
      실패 목록에 있는 id만 담습니다. 이때 '없는 id = 통과'로 간주합니다.
    """
    if isinstance(run.get("results"), list):
        return {str(r["id"]): r.get("status", FAIL) for r in run["results"] if "id" in r}
    return {str(f["id"]): f.get("status", FAIL) for f in run.get("failures", []) if "id" in f}


def compare_eval(before: dict, after: dict) -> dict:
    """개선 전후를 비교합니다. LLM을 호출하지 않습니다.

    통과율 총합만 보면 '무엇이 좋아지고 무엇이 깨졌는지'를 알 수 없습니다.
    한 유형을 고치면 다른 유형이 깨지는 일이 실제로 자주 일어납니다.

    Args:
        before, after: run_eval이 반환한 dict

    Returns:
        {
          "fixed":     [before에서 실패했다가 after에서 통과한 문항 id],
          "regressed": [before에서 통과했다가 after에서 실패한 문항 id],
          "delta": after.passed - before.passed,
          "safe": regressed가 비었으면 True,
        }

    추가 반환 키 (판단을 돕는 참고 정보. safe 판정에는 쓰지 않습니다):
      - "added"   : after에만 있는 문항 id (평가셋에 새로 넣은 문항)
      - "removed" : before에만 있는 문항 id (평가셋에서 빠진 문항)
      - "by_type_delta": {유형: after 통과 수 - before 통과 수}

    판정 규칙
      - 상태가 PASS가 아니면(FAIL, ERROR 모두) '실패'로 봅니다.
        통과하던 문항이 예외로 터지게 된 것(pass → error)도 회귀입니다.
      - 양쪽에 모두 있는 문항만 fixed/regressed 대상입니다. 새로 넣은 문항이
        실패하는 것은 회귀가 아니라 '새로 드러난 빈틈'이라 added로 따로 보여 줍니다.
      - 목록 순서는 before 평가셋 순서를 따릅니다. 같은 입력이면 항상 같은 출력입니다.
    """
    b_map, a_map = _status_map(before), _status_map(after)
    full = isinstance(before.get("results"), list) and isinstance(after.get("results"), list)

    fixed: list[str] = []
    regressed: list[str] = []
    if full:
        for qid, b_status in b_map.items():
            if qid not in a_map:
                continue
            b_ok, a_ok = b_status == PASS, a_map[qid] == PASS
            if not b_ok and a_ok:
                fixed.append(qid)
            elif b_ok and not a_ok:
                regressed.append(qid)
        added = [qid for qid in a_map if qid not in b_map]
        removed = [qid for qid in b_map if qid not in a_map]
    else:
        # failures만 있는 경우: 같은 평가셋을 돌렸다고 가정하고 실패 집합의 차로 계산합니다.
        fixed = [qid for qid in b_map if qid not in a_map]
        regressed = [qid for qid in a_map if qid not in b_map]
        added, removed = [], []

    b_types, a_types = before.get("by_type", {}) or {}, after.get("by_type", {}) or {}
    by_type_delta = {
        t: (a_types.get(t, {}).get("passed", 0) - b_types.get(t, {}).get("passed", 0))
        for t in list(dict.fromkeys([*b_types, *a_types]))
    }

    return {
        "fixed": fixed,
        "regressed": regressed,
        "delta": int(after.get("passed", 0)) - int(before.get("passed", 0)),
        # 회귀가 하나라도 있으면 통과 수가 늘었어도 안전하지 않습니다.
        # 반대로 회귀가 없으면 반드시 True — 늘 경고하는 게이트는 아무도 읽지 않습니다.
        "safe": not regressed,
        "added": added,
        "removed": removed,
        "by_type_delta": by_type_delta,
    }


# ══════════════════════════════════════════════════════════════════
# 직접 실행용 데모 — LLM 없이 저장소 루트 eval_set.json 으로 흐름을 보여 줍니다
# (채점기는 이 아래를 실행하지 않습니다. `python submission.py` 로 직접 실행할 때만 동작)
# 실제 시스템을 잴 때는 answer_fn 자리에 6일차 Supervisor 호출 함수를,
# judge 자리에 day7_practice/run_eval.py 의 유형별 판정 규칙을 넣으면 됩니다.
# ══════════════════════════════════════════════════════════════════

def _demo() -> None:
    root = Path(__file__).resolve().parents[3]
    eval_set = json.loads((root / "eval_set.json").read_text(encoding="utf-8"))
    tracer = TraceCollector()

    def make_answer_fn(broken_type: str):
        """broken_type 유형의 질문에서 일부러 실패하는 가짜 시스템."""
        type_of = {i["question"]: i["type"] for i in eval_set}

        def answer_fn(question: str) -> str:
            tracer.record("llm_start", "fake-llm")
            tracer.record("llm_end", "fake-llm", tokens={"input": len(question), "output": 3})
            if type_of.get(question) == "tool_recovery":
                tracer.record("tool_start", "get_asset_status")
                tracer.record("tool_end", "get_asset_status")
                raise TimeoutError("백엔드 타임아웃 (데모용)")
            return "" if type_of.get(question) == broken_type else "답변"
        return answer_fn

    before = run_eval(make_answer_fn("no_answer"), eval_set)
    after = run_eval(make_answer_fn("guardrail"), eval_set)
    diff = compare_eval(before, after)

    for label, r in (("before", before), ("after", after)):
        print(f"[{label}] {r['passed']}/{r['total']}  "
              + ", ".join(f"{t} {v['passed']}/{v['total']}" for t, v in r["by_type"].items()))
    print("fixed    :", diff["fixed"])
    print("regressed:", diff["regressed"])
    print("delta    :", diff["delta"], "/ safe:", diff["safe"])
    print("trace    :", tracer.summary()["counts"])


if __name__ == "__main__":
    _demo()
