"""Day 3 — 장애 원인 조사 Agent.

도구 세 개로 장애 원인을 좁혀 갑니다.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. 모델은 함수 안에서 만듭니다. `llm` 인자가 오면 그것을 씁니다.
  3. 도구는 결정적이어야 합니다. 같은 인자에 항상 같은 결과를 돌려주세요.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"

# 교육용 더미 데이터입니다. (practice/data/ops_fixtures.json)
DATA = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "ops_fixtures.json").read_text(
        encoding="utf-8"
    )
)


def _default_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


def get_text(message) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content or ""


def _known_services(section: str) -> str:
    """DATA의 한 구역(deploys/metrics/logs)에서 조회 가능한 서비스 이름 목록을 문자열로 만듭니다.

    sorted로 순서를 고정해 결과가 항상 같게 만듭니다(결정성).
    """
    if section == "logs":
        names = {key.split("@", 1)[0] for key in DATA["logs"]}
    else:
        names = set(DATA[section])
    return ", ".join(sorted(names))


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# ---------- 1) 도구 인자 스키마 ----------
# 모델은 이 description을 보고 인자에 무엇을 넣을지 정합니다.
# args_schema가 있으면 모델이 인자 이름을 지어내는 것을 한 겹 앞에서 막아 줍니다.

class DeployArgs(BaseModel):
    service: str = Field(
        ...,
        description="배포 이력을 조회할 서비스 이름. 예: 'payment-api', 'auth-service'",
    )


class LogsArgs(BaseModel):
    service: str = Field(
        ...,
        description="로그를 조회할 서비스 이름. 예: 'payment-api'",
    )
    since: Optional[str] = Field(
        default=None,
        description=(
            "로그를 볼 시점. get_deploy_history 결과의 배포 시각(ISO 8601 문자열)을 "
            "그대로 넣으세요. 예: '2026-08-11T02:58:00+09:00'. "
            "비워 두면 평상시(기본) 로그를 돌려줍니다."
        ),
    )


class MetricsArgs(BaseModel):
    service: str = Field(
        ...,
        description="현재 지표(에러율, p99 지연, CPU)를 조회할 서비스 이름. 예: 'payment-api'",
    )


# ---------- 2) 도구 정의 ----------
# 모델이 도구를 고를 때 보는 것은 이름과 docstring뿐입니다.
# 그래서 '언제 쓰는지'와 '언제 쓰지 말아야 하는지'를 함께 적습니다.

@tool(args_schema=DeployArgs)
def get_deploy_history(service: str) -> str:
    """서비스의 최근 배포 이력(버전, 배포 시각, 배포 팀)을 최신순으로 조회합니다.

    장애 조사를 시작할 때 가장 먼저 사용해 '최근 배포가 있었는지, 언제였는지'를 확인하세요.
    여기서 얻은 배포 시각은 get_logs의 since 인자로 넘깁니다.
    로그 내용이나 에러율 같은 지표를 보려는 용도로는 쓰지 마세요.
    """
    service = (service or "").strip()
    if not service:
        return (
            "서비스 이름이 비어 있습니다. 조회할 서비스 이름을 넣어 주세요. "
            f"조회 가능한 서비스: {_known_services('deploys')}"
        )

    deploys = DATA["deploys"].get(service)
    if not deploys:
        return (
            f"'{service}' 서비스의 배포 이력이 없습니다. "
            f"조회 가능한 서비스: {_known_services('deploys')}"
        )

    # 최신순으로 정렬합니다. (ISO 8601 문자열은 같은 시간대면 문자열 정렬 = 시간 정렬)
    ordered = sorted(deploys, key=lambda d: d["at"], reverse=True)
    lines = [f"[{service}] 배포 이력 {len(ordered)}건 (최신순)"]
    for d in ordered:
        lines.append(f"- {d['version']} | 배포 시각 {d['at']} | 배포 팀 {d['by']}")
    lines.append(
        f"가장 최근 배포 시각은 {ordered[0]['at']} 입니다. "
        "그 시점 로그를 보려면 get_logs의 since에 이 값을 그대로 넣으세요."
    )
    return "\n".join(lines)


@tool(args_schema=LogsArgs)
def get_logs(service: str, since: Optional[str] = None) -> str:
    """서비스의 로그를 조회합니다. since에 배포 시각을 넣으면 그 시점 전후의 로그를 봅니다.

    에러 메시지나 예외처럼 '무엇이 찍혔는지'를 확인할 때 사용하세요.
    배포 직후 로그를 보려면 먼저 get_deploy_history로 배포 시각을 알아낸 뒤 since로 넘기세요.
    배포 이력이나 수치 지표를 보려는 용도로는 쓰지 마세요.
    """
    service = (service or "").strip()
    if not service:
        return (
            "서비스 이름이 비어 있습니다. 조회할 서비스 이름을 넣어 주세요. "
            f"조회 가능한 서비스: {_known_services('logs')}"
        )

    logs = DATA["logs"]
    since = (since or "").strip() or None

    if since:
        key = f"{service}@{since}"
        if key in logs:
            return f"[{service}] {since} 시점 로그\n{logs[key]}"

    default_key = f"{service}@default"
    if default_key not in logs:
        return (
            f"'{service}' 서비스의 로그가 없습니다. "
            f"조회 가능한 서비스: {_known_services('logs')}"
        )

    if since:
        # since를 줬지만 그 시점 로그가 없으면, 사실을 알리고 기본 로그를 돌려줍니다.
        return (
            f"[{service}] {since} 시점 로그는 없습니다. "
            "get_deploy_history의 배포 시각을 그대로 넣었는지 확인하세요.\n"
            f"대신 평상시(기본) 로그를 보여 드립니다.\n{logs[default_key]}"
        )
    return (
        f"[{service}] 평상시(기본) 로그\n{logs[default_key]}\n"
        "(특정 배포 시점 로그를 보려면 since에 배포 시각을 넣으세요.)"
    )


@tool(args_schema=MetricsArgs)
def get_metrics(service: str) -> str:
    """서비스의 현재 핵심 지표(5분 에러율, p99 지연시간, CPU 사용률)를 조회합니다.

    장애의 규모나 영향(에러가 얼마나 늘었는지, 느려졌는지)을 수치로 확인할 때 사용하세요.
    배포 시각이나 에러 메시지 원문을 찾으려는 용도로는 쓰지 마세요.
    """
    service = (service or "").strip()
    if not service:
        return (
            "서비스 이름이 비어 있습니다. 조회할 서비스 이름을 넣어 주세요. "
            f"조회 가능한 서비스: {_known_services('metrics')}"
        )

    m = DATA["metrics"].get(service)
    if m is None:
        return (
            f"'{service}' 서비스의 지표가 없습니다. "
            f"조회 가능한 서비스: {_known_services('metrics')}"
        )

    return (
        f"[{service}] 현재 지표\n"
        f"- 5분 에러율: {m['error_rate_5m'] * 100:.1f}%\n"
        f"- p99 지연시간: {m['p99_latency_ms']}ms\n"
        f"- CPU 사용률: {m['cpu'] * 100:.0f}%"
    )


# ---------- 3) 도구 모음 ----------
TOOLS: list = [get_deploy_history, get_logs, get_metrics]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}   # 이름으로 도구를 찾기 위한 사전


# ---------- 4) 그래프 State ----------
class AgentState(TypedDict):
    # [수동 루프의 messages 리스트] -> add_messages reducer가 알아서 누적합니다.
    messages: Annotated[list, add_messages]
    # 지금까지 실제로 실행한 도구 호출 횟수 (심화: 예산 판정에 사용)
    tool_calls_made: int
    # 조사를 멈춘 이유 (멈추지 않았으면 빈 문자열)
    stop_reason: str


SYSTEM_PROMPT = """너는 장애 원인을 조사하는 운영(SRE) 어시스턴트야.

[조사 순서]
1. 먼저 get_deploy_history로 최근 배포가 있었는지, 배포 시각이 언제인지 확인해.
2. 그 배포 시각을 get_logs의 since 인자에 그대로 넣어 그 시점 로그를 확인해.
3. get_metrics로 에러율·지연·CPU 같은 지표가 어떻게 움직였는지 확인해.

[규칙]
- 직전 도구 결과를 보고 다음에 어떤 도구를 부를지 매번 다시 판단해.
- 같은 도구를 같은 인자로 다시 부르지 마.
- 도구 결과에 없는 사실은 지어내지 마.
- 장애 조사와 무관한 인사나 일반 질문에는 도구 없이 바로 답해.
- 결론에는 의심되는 원인, 근거(배포/로그/지표), 권장 조치를 짧게 정리해."""


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 한 번의 조사에서 허용할 도구 호출 상한 (3 이상 8 이하)
# 세 도구를 한 번씩 쓰는 데 3회, 재조회 여유를 조금 더 둡니다.
MAX_TOOL_CALLS = 6

REQUIRED_TOOLS = {"get_deploy_history", "get_logs", "get_metrics"}


def _iter_tool_calls(messages: list):
    """메시지 목록에서 AI가 요청한 도구 호출을 (이름, 인자) 순서대로 꺼냅니다."""
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if isinstance(tc, dict):
                yield tc.get("name"), tc.get("args") or {}
            else:
                yield getattr(tc, "name", None), getattr(tc, "args", None) or {}


def should_stop(messages: list, tool_calls_made: int) -> tuple[bool, str]:
    """더 조사할지 멈출지 결정합니다. LLM을 호출하지 않습니다.

    판정 규칙:
      - tool_calls_made >= MAX_TOOL_CALLS        -> (True, "예산 초과")
      - 같은 도구를 같은 인자로 2회 이상 호출함  -> (True, "중복 호출")
      - 세 도구를 모두 한 번 이상 호출함         -> (True, "근거 충분")
      - 그 외                                    -> (False, 아무 문자열)

    Args:
        messages: 지금까지의 메시지 목록. 도구 호출 기록이 들어 있습니다.
        tool_calls_made: 지금까지 실행한 도구 호출 횟수

    Returns:
        (중단할지 여부, 이유 문자열)
    """
    # 1) 예산: 비용의 상한선
    if tool_calls_made >= MAX_TOOL_CALLS:
        return True, "예산 초과"

    # 2) 중복 호출: 같은 질문을 반복하는 루프에서 빠져나오기
    #    인자 dict는 키 순서가 달라도 같은 호출로 보도록 정렬해 문자열로 만듭니다.
    seen: set[tuple[str, str]] = set()
    called: set[str] = set()
    for name, args in _iter_tool_calls(messages):
        key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))
        if key in seen:
            return True, "중복 호출"
        seen.add(key)
        called.add(name)

    # 3) 근거 충분: 세 도구를 모두 썼으면 더 파고들지 않기
    if REQUIRED_TOOLS <= called:
        return True, "근거 충분"

    return False, "조사 계속"


# ══════════════════════════════════════════════════════════════════
# 그래프 조립
# ══════════════════════════════════════════════════════════════════

def build_agent(llm=None):
    """도구를 쓸 수 있는 Agent 그래프를 만들어 컴파일해 반환합니다.

    반드시 지킬 것:
      - agent 노드와 tools 노드 사이에 순환이 있어야 합니다.
      - 탈출 조건이 있어야 합니다. 없으면 무한 루프입니다.

    create_react_agent를 써도 되고 StateGraph로 직접 만들어도 됩니다.
    심화 파트를 구현했다면 should_stop 판정을 그래프에 연결하세요.

    그래프 모양:

        START -> agent --(도구 요청 없음)--> END
                   |
                (도구 요청)
                   v
                 tools --(should_stop == False)--> agent   (순환)
                   |
             (should_stop == True)
                   v
                finalize -> END
    """
    llm = llm or _default_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    # ---------- 노드 1) agent: 모델이 다음 행동을 정합니다 ----------
    # [수동 루프의 invoke와 append]
    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    # ---------- 노드 2) tools: 요청된 도구를 실행하고 ToolMessage로 포장합니다 ----------
    # 내장 ToolNode 대신 직접 실행하는 이유: 한 번에 여러 도구를 요청해도
    # 남은 예산만큼만 실행해 MAX_TOOL_CALLS를 절대 넘지 않게 하려는 것입니다.
    def tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        made = state.get("tool_calls_made", 0)
        results: list[ToolMessage] = []

        for tc in last.tool_calls:
            name, args, call_id = tc["name"], tc.get("args") or {}, tc["id"]

            if made >= MAX_TOOL_CALLS:
                # 모든 tool_call에는 짝이 되는 ToolMessage가 있어야 하므로 실행 거부도 메시지로 남깁니다.
                content = f"도구 호출 예산({MAX_TOOL_CALLS}회)을 모두 써서 실행하지 않았습니다."
                results.append(ToolMessage(content=content, tool_call_id=call_id, name=name, status="error"))
                continue

            fn = TOOLS_BY_NAME.get(name)
            if fn is None:
                content = f"'{name}' 도구는 없습니다. 사용 가능한 도구: {sorted(TOOLS_BY_NAME)}"
                results.append(ToolMessage(content=content, tool_call_id=call_id, name=name, status="error"))
                continue

            try:
                content = str(fn.invoke(args))
                status = "success"
            except Exception as exc:  # noqa: BLE001 - 인자 오류 등을 모델이 읽고 고칠 수 있게 돌려줍니다.
                content = f"도구 실행 실패: {type(exc).__name__}: {exc}. 인자를 확인하세요."
                status = "error"
            made += 1
            results.append(ToolMessage(content=content, tool_call_id=call_id, name=name, status=status))

        return {"messages": results, "tool_calls_made": made}

    # ---------- 노드 3) finalize: 조사를 멈추고 지금까지의 근거로 결론을 냅니다 ----------
    def finalize(state: AgentState) -> dict:
        _, reason = should_stop(state["messages"], state.get("tool_calls_made", 0))
        system = (
            f"{SYSTEM_PROMPT}\n\n"
            f"[조사 종료] 이유: {reason}. 더 이상 도구를 호출하지 말고, "
            "지금까지의 도구 결과만으로 원인·근거·권장 조치를 정리해."
        )
        response = llm_with_tools.invoke([SystemMessage(content=system)] + state["messages"])
        # 모델이 그래도 도구를 요청했다면 무시하고 텍스트만 최종 답변으로 남깁니다.
        final = AIMessage(content=get_text(response) or f"조사를 종료합니다. (이유: {reason})")
        return {"messages": [final], "stop_reason": reason}

    # ---------- 라우팅 1) agent 다음: 도구 요청이 없으면 종료 ----------
    # [수동 루프의 if not ai_message.tool_calls]
    def route_after_agent(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    # ---------- 라우팅 2) tools 다음: should_stop 판정으로 탈출 여부 결정 ----------
    def route_after_tools(state: AgentState) -> Literal["agent", "finalize"]:
        stop, _ = should_stop(state["messages"], state.get("tool_calls_made", 0))
        return "finalize" if stop else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    builder.add_conditional_edges(
        "tools", route_after_tools, {"agent": "agent", "finalize": "finalize"}  # 순환 + 탈출
    )
    builder.add_edge("finalize", END)

    return builder.compile()


if __name__ == "__main__":
    # 실제 Bedrock 모델로 한 번 돌려 봅니다. (채점 테스트는 이 부분을 실행하지 않습니다)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from langchain_core.messages import HumanMessage

    graph = build_agent()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="payment-api에서 5xx가 급증했습니다. 원인을 조사해 주세요.")],
            "tool_calls_made": 0,
            "stop_reason": "",
        },
        {"recursion_limit": 30},
    )
    for m in result["messages"]:
        label = type(m).__name__
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"  [{label}] 도구 요청: {tc['name']}({tc['args']})")
        elif m.content:
            print(f"  [{label}] {get_text(m)[:300]}")
    print(f"\n도구 호출 {result.get('tool_calls_made', 0)}회 / 종료 이유: {result.get('stop_reason') or '모델이 스스로 종료'}")
