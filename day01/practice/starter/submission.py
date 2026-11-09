"""Day 1 — 운영 알람 트리아지 체인.

알람 하나를 넣으면 구조화된 분류 결과와 담당팀 안내 초안을 동시에 만듭니다.

[중요] 이 파일의 규칙
  1. 함수 이름과 인자를 바꾸지 마세요. 채점기가 이 이름으로 찾습니다.
  2. 모델은 반드시 함수 **안에서** 만듭니다. 모듈 최상단에서 만들면
     채점기가 가짜 모델을 넣을 수 없어 채점이 실패합니다.
  3. `llm` 인자가 들어오면 그것을 쓰고, None이면 직접 만듭니다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 과정 전체에서 고정으로 쓰는 값입니다. 바꾸지 마세요.
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"


def _default_llm():
    """기본 모델을 만듭니다. 채점기는 이 함수를 부르지 않습니다.

    import를 함수 안에 둔 이유: 모듈을 불러오는 것만으로 AWS 자격증명을
    요구하지 않게 하기 위해서입니다.
    """
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


def get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# TODO: 분류 결과 스키마를 정의합니다.
#   - 필드 4개 이상
#   - 모든 필드에 description (모델이 보는 것은 이 설명문뿐입니다)
#   - severity는 Literal 또는 Enum으로 값을 제한합니다
#
# 아래는 최소 형태의 예시입니다. 본인 업무에 맞게 필드를 바꾸고 늘리세요.
class TriageResult(BaseModel):
    """고객 문의 분류 결과입니다."""
    severity: Literal["P1", "P2", "P3", "P4"] = Field(description="긴급도: P1(통관 보류·선적 지연 등 즉시 대응)~P4(일반 정보 문의) 중 하나")
    mode: str = Field(description="모드: 'FCL', 'LCL', '항공', '특송' 중 하나. 문의하는 국제운송 모드에 따라 나눌 것")
    type: str = Field(description="수출입: '수출', '수입', '3국간' 중 하나. 대한민국을 기준으로 판단할 것")
    intend : str = Field(description="의도: '견적 요청', '서비스 문의', '플랫폼 사용법 문의', '물류지식 문의' 중 하나. 문의 내용 중 가장 비중이 높은 의도 한가지를 찾아서 판단할 것")
    ecomm : str = Field(description="이커머스: 'O', 'X' 중 하나. 아마존, FBA, 틱톡, FBT 등 이커머스 플랫폼을 의미하는 단어가 포함되어 있으면 O로, 아니면 X로 선택")


TRIAGE_SCHEMA = TriageResult

# 상담원 안내(답변) 체인의 system 메시지입니다. 첼로스퀘어 고객 문의 담당자 역할을 지정합니다.
NOTICE_SYSTEM_PROMPT = (
    "너는 첼로스퀘어의 고객 문의 담당자야. "
    "고객이 국제운송과 관련된 질문을 하면 받아서 기준에 맞춰서 분류해 줘야 해. "
    "그 분류를 모두 보여주고, 질문에 맞는 답변도 5문장 내로 만들어서 보여줘. "
    "친절하면서 전문적인 말투를 사용하도록 해."
)


def _to_text(inquiry: Any) -> str:
    """체인 입력(dict 또는 문자열)을 모델에 넣을 문의 본문 문자열로 바꿉니다.

    - {"inquiry": "..."} 형태면 문의 본문만 꺼냅니다.
    - 그 밖의 dict는 "키: 값" 줄로 풀어서 넘깁니다. (키가 빠져 있어도 깨지지 않도록)
    - 그 외 타입은 문자열로 변환합니다.
    """
    if isinstance(inquiry, dict):
        if "inquiry" in inquiry:
            return str(inquiry["inquiry"])
        return "\n".join(f"{k}: {v}" for k, v in inquiry.items())
    return str(inquiry)


def _fallback_triage(raw_text: str) -> dict:
    """모델 응답이 JSON 형식을 깨뜨렸을 때 돌려줄 기본 분류 결과입니다.

    스키마의 모든 필드를 '판단불가'로 채우고, 원문 응답과 실패 표시를 남겨
    상담원이 수동으로 다시 분류할 수 있게 합니다.
    """
    result = {name: "판단불가" for name in TRIAGE_SCHEMA.model_fields}
    result["parse_error"] = True
    result["raw_output"] = raw_text
    return result


def build_triage_chain(llm=None):
    """고객 문의 dict를 받아 TRIAGE_SCHEMA 형태의 dict를 반환하는 체인을 만듭니다.

    반드시 지킬 것:
      - 프롬프트에 파서의 format_instructions를 실제로 주입해야 합니다.
        스키마만 정의하고 프롬프트에 넣지 않으면 모델은 형식을 모릅니다.
    """
    from langchain_core.exceptions import OutputParserException
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda

    llm = llm or _default_llm()

    # 스키마(TriageResult)를 기반으로 JSON 출력 형식 지침을 만들어 주는 파서입니다.
    parser = JsonOutputParser(pydantic_object=TRIAGE_SCHEMA)

    prompt = PromptTemplate(
        template=(
            "너는 첼로스퀘어의 고객 문의 담당자야.\n"
            "다음 국제운송 고객 문의를 읽고 운송 모드, 수출입 구분, 문의 의도, "
            "이커머스 여부를 각 필드 설명의 기준에 맞춰 분류해.\n\n"
            "고객 문의:\n{inquiry}\n\n"
            "{format_instructions}\n"
        ),
        input_variables=["inquiry"],
        # 형식 지침을 프롬프트에 실제로 주입합니다.
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    def safe_parse(message: Any) -> dict:
        """모델 응답을 JSON으로 파싱하고, 실패하면 기본값 dict로 복구합니다."""
        text = get_text(message)
        try:
            parsed = parser.parse(text)
        except OutputParserException:
            return _fallback_triage(text)
        # JSON이 dict가 아닌 값(리스트, 숫자 등)으로 나와도 dict 계약을 지킵니다.
        if not isinstance(parsed, dict):
            return _fallback_triage(text)
        return parsed

    return {"inquiry": RunnableLambda(_to_text)} | prompt | llm | RunnableLambda(safe_parse)


def build_notice_chain(llm=None):
    """고객 문의 dict를 받아 상담원 안내 초안 문자열을 반환하는 체인을 만듭니다.

    반드시 지킬 것:
      - system 메시지로 역할을 지정합니다 (30자 이상).
      - 기본 예시 문구를 그대로 두지 말고 본인 조직의 맥락으로 바꾸세요.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    llm = llm or _default_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NOTICE_SYSTEM_PROMPT),
            # 문의 본문은 변수로 넘기므로 본문에 중괄호가 있어도 템플릿이 깨지지 않습니다.
            ("human", "{inquiry}"),
        ]
    )

    return {"inquiry": RunnableLambda(_to_text)} | prompt | llm | StrOutputParser()


def build_full_chain(llm=None):
    """두 체인을 병렬로 묶어 {"triage": dict, "notice": str} 를 반환합니다.

    심화 파트를 구현했다면, triage 결과는 apply_severity_rules를 거친 값이어야 합니다.
    """
    from langchain_core.runnables import RunnableLambda, RunnableParallel

    llm = llm or _default_llm()

    triage_chain = build_triage_chain(llm)
    notice_chain = build_notice_chain(llm)

    # 1단계: 원본 입력과 모델 분류 결과를 함께 들고 갑니다. (규칙 적용에 원본이 필요)
    # 2단계: 결정적 규칙(apply_severity_rules)을 얹습니다.
    triage_with_rules = RunnableParallel(
        raw=triage_chain,
        inquiry=RunnableLambda(lambda x: x),
    ) | RunnableLambda(
        lambda d: apply_severity_rules(
            d["raw"], d["inquiry"] if isinstance(d["inquiry"], dict) else {}
        )
    )

    return RunnableParallel(triage=triage_with_rules, notice=notice_chain)


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 장애 시 최우선으로 봐야 하는 서비스입니다.
CRITICAL_SERVICES: set[str] = {"booking-api", "payment-api", "quote-api", "tracking-api"}

# severity 등급의 높낮이 비교용입니다. 숫자가 작을수록 급합니다.
SEVERITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


def apply_severity_rules(triage: dict, alert: dict) -> dict:
    """모델이 매긴 severity 위에 결정적 규칙을 얹습니다.

    이 함수는 LLM을 호출하지 않습니다. 같은 입력에 항상 같은 출력을 내야 합니다.

    규칙:
      - alert["service"]가 CRITICAL_SERVICES에 있으면 최소 P2
      - alert["message"]에 "5xx" 또는 "outage"가 있으면 최소 P1
      - 둘 다면 더 높은 쪽(숫자가 작은 쪽)을 적용
      - 아무것도 해당 없으면 모델 판단을 그대로 둠
      - 규칙이 적용됐으면 그 사실을 결과에 남길 것 (예: rule_applied 키)

    Returns:
        규칙이 반영된 새 dict. 원본을 그대로 바꾸지 말고 복사해서 쓰는 편이 안전합니다.
    """
    result = dict(triage)  # 원본을 바꾸지 않도록 복사합니다.
    alert = alert or {}

    service = str(alert.get("service", "") or "")
    message = str(alert.get("message", "") or "").lower()

    # 규칙별로 요구되는 '최소 등급'을 모읍니다.
    floors: list[tuple[str, str]] = []
    if service in CRITICAL_SERVICES:
        floors.append(("P2", f"critical_service:{service}"))
    if "5xx" in message or "outage" in message:
        floors.append(("P1", "message_keyword:5xx/outage"))

    if not floors:
        # 해당 규칙이 없으면 모델 판단을 그대로 둡니다.
        return result

    # 둘 다 해당하면 더 급한 쪽(숫자가 작은 쪽)을 적용합니다.
    floor_sev, _ = min(floors, key=lambda f: SEVERITY_ORDER[f[0]])
    current = result.get("severity")
    current_rank = SEVERITY_ORDER.get(current, 99)  # 값이 없거나 이상하면 가장 낮은 등급 취급

    if SEVERITY_ORDER[floor_sev] < current_rank:
        # 규칙 최소 등급이 모델 판단보다 급할 때만 올립니다. (이미 높으면 낮추지 않음)
        result["severity"] = floor_sev
        result["rule_applied"] = [reason for _, reason in floors]
        result["severity_before_rules"] = current

    return result


# ══════════════════════════════════════════════════════════════════
# 실행 예시
#   python my_inquiry_chain.py
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # 교육용 더미 데이터입니다.
    sample = {
        "inquiry": "부산에서 미국 LA 아마존 FBA 창고로 20피트 컨테이너 1대 보내려고 하는데 "
        "운임 견적 받을 수 있을까요?"
    }

    out = build_full_chain().invoke(sample)
    triage = out["triage"]

    print("모드:", triage.get("mode"))
    print("수출입:", triage.get("type"))
    print("의도:", triage.get("intend"))
    print("이커머스:", triage.get("ecomm"))
    print("\n답변 초안:")
    print(out["notice"])
