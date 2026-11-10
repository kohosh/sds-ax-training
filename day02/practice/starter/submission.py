"""Day 2 — 런북 RAG와 평가셋.

문서를 인덱싱하고, 질문에 근거와 함께 답합니다.

[중요] 이 파일의 규칙
  1. 함수 이름과 인자를 바꾸지 마세요.
  2. 모델·임베딩은 반드시 함수 **안에서** 만듭니다.
  3. `llm` 인자가 들어오면 그것을 씁니다. 채점기가 가짜 모델을 넣습니다.
  4. `build_retriever`와 `build_rag_chain`은 **주어진 vectorstore/retriever를 그대로 씁니다.**
     함수 안에서 새로 인덱싱하지 마세요.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import PureWindowsPath
from typing import Any

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"

PERSIST_DIR = "C:/Users/SDS/Desktop/sds-ax-practice/day2_practice/chroma_db"
COLLECTION = "sds_policies"


def _default_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


def _default_embeddings():
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=REGION)


def get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def format_docs(docs) -> str:
    """검색된 문서를 프롬프트에 넣을 문자열로 만듭니다."""
    return "\n\n".join(
        f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}" for d in docs
    )


def _file_name(path: Any) -> str:
    """경로에서 파일명만 뽑습니다. (예: 'docs/leave_policy.md' -> 'leave_policy.md')

    PureWindowsPath는 '/'와 '\\'를 모두 경로 구분자로 인식하므로
    윈도우 경로든 리눅스 경로든 같은 결과를 냅니다.
    """
    return PureWindowsPath(str(path)).name


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# 파일별 업무 메타데이터 (교육용 더미 데이터입니다)
FILE_META = {
    "C:/Users/SDS/Desktop/sds-ax-practice/sds-ax-training/sds-ax-training/day02/practice/docs/leave_policy.md":       {"department": "관리팀", "updated_at": "2026-09-01"},
    "C:/Users/SDS/Desktop/sds-ax-practice/sds-ax-training/sds-ax-training/day02/practice/docs/remote_work_policy.md": {"department": "혁신팀", "updated_at": "2026-10-02"},
    "C:/Users/SDS/Desktop/sds-ax-practice/sds-ax-training/sds-ax-training/day02/practice/docs/travel_policy.md":      {"department": "재무팀", "updated_at": "2025-11-03"},
}

# FILE_META에 없는 파일에 붙일 담당 조직 기본값입니다.
DEFAULT_DEPARTMENT = "미지정"

# 청크 최대 길이(글자 수). 실습 문서의 섹션 하나가 대략 150~450자이므로
# 500이면 섹션 대부분이 한 청크로 유지되고, 긴 섹션만 한 번 더 나뉩니다.
# README 권장(800 이하)도 지킵니다.
CHUNK_SIZE = 500
# 앞 청크 끝 50자를 다음 청크 앞에 겹쳐 넣어, 경계에서 문장이 끊겨 뜻이 사라지는 것을 줄입니다.
CHUNK_OVERLAP = 50


def _lookup_meta(path: str) -> dict:
    """파일 경로에 맞는 department / updated_at 을 찾아 돌려줍니다.

    채점 PC는 폴더 위치가 다르므로 전체 경로가 아니라 '파일명'으로 비교합니다.
    FILE_META에 없는 파일은 담당 조직을 '미지정'으로, 수정일은 파일의 실제
    마지막 수정 날짜(YYYY-MM-DD)로 채웁니다. 세 키가 빠지는 청크가 없게 하기 위함입니다.
    """
    meta_by_name = {_file_name(p): m for p, m in FILE_META.items()}
    found = meta_by_name.get(_file_name(path))
    if found:
        return dict(found)

    try:
        updated_at = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except OSError:
        updated_at = datetime.now().strftime("%Y-%m-%d")
    return {"department": DEFAULT_DEPARTMENT, "updated_at": updated_at}


def _load_documents(doc_paths: list[str]) -> list[Document]:
    """경로 목록의 파일을 읽어 Document 리스트로 만듭니다.

    경로 목록이 비어 있으면 실습 docs 폴더 전체를 DirectoryLoader로 읽습니다.
    (모듈 최상단에서 읽으면 파일을 import하는 순간 실행되어 채점 PC에서 오류가 나므로
    함수 안으로 옮겼습니다.)
    """
    if doc_paths:
        docs: list[Document] = []
        for path in doc_paths:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        return docs

    dir_loader = DirectoryLoader(
        "C:/Users/SDS/Desktop/sds-ax-practice/sds-ax-training/sds-ax-training/day02/practice/docs",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return dir_loader.load()


def build_chunks(doc_paths: list[str]) -> list:
    """문서를 읽어 청크 Document 리스트를 만듭니다.

    반드시 지킬 것:
      - 모든 청크 metadata에 source / department / updated_at 세 키가 있어야 합니다.
      - 메타데이터는 **분할하기 전에** 붙여야 상속됩니다.
      - 고정 길이로 자르지 말고 RecursiveCharacterTextSplitter 또는
        MarkdownHeaderTextSplitter를 쓰세요.

    Args:
        doc_paths: 읽을 마크다운 파일 경로 목록

    Returns:
        langchain_core.documents.Document 의 리스트
    """
    # 1단계 분할기: 마크다운 제목(#, ##)을 기준으로 섹션 단위로 자릅니다.
    #   strip_headers=False -> '## 2. 이월' 같은 제목 줄을 본문에 남겨 원문을 보존합니다.
    headers = [("#", "title"), ("##", "section")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)

    # 2단계 분할기: 섹션이 CHUNK_SIZE보다 길면 문단 -> 줄 -> 문장 순으로 한 번 더 나눕니다.
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks: list[Document] = []
    for doc in _load_documents(doc_paths):
        # 1) 분할하기 "전에" 원본 Document에 메타데이터 3키를 붙입니다.
        original_path = doc.metadata.get("source", "")
        doc.metadata.update(_lookup_meta(original_path))
        doc.metadata["source"] = _file_name(original_path)  # source = 원본 파일명

        # 2-1) 제목 기준 분할. MarkdownHeaderTextSplitter는 문자열만 받아서
        #      원본 메타데이터를 모르기 때문에, 원본 메타데이터를 각 섹션에 복사해 상속시킵니다.
        sections = [
            Document(
                page_content=section.page_content,
                metadata={**doc.metadata, **section.metadata},
            )
            for section in md_splitter.split_text(doc.page_content)
        ]

        # 2-2) 길이 기준 분할. split_documents는 메타데이터를 자동으로 상속합니다.
        chunks.extend(size_splitter.split_documents(sections))

    return chunks


def build_retriever(vectorstore):
    """주어진 벡터저장소에서 retriever를 만듭니다.

    여기서 새로 인덱싱하지 마세요. 인자로 받은 vectorstore를 그대로 씁니다.
    k 값은 직접 정하고, 왜 그 값인지 주석으로 남기세요.
    """
    # k=3 을 고른 이유:
    #   - 한 질문의 답은 보통 섹션 1개(청크 1~2개)에 들어 있으므로 3개면 정답 청크를 놓칠 위험이 작습니다.
    #   - 문서가 3개뿐이라 k를 더 키우면 무관한 런북 청크가 컨텍스트를 채워 모델이 그쪽으로 끌려갑니다.
    #   - 프롬프트 길이(=비용·지연)도 작게 유지됩니다. (day2_practice/retriever.py의 k 비교 실습 기준)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


RAG_SYSTEM_PROMPT = (
    "너는 사내 문서 안내 어시스턴트야. 아래 '검색된 문서'에 적힌 내용만 근거로 한국어로 간결하게 답해.\n"
    "문서에 답이 없으면 추측하지 말고 다른 말 없이 정확히 "
    "'제공된 문서에서 답을 찾을 수 없습니다.' 라고만 답해.\n\n"
    "--- 검색된 문서 ---\n{context}"
)


def build_rag_chain(retriever, llm=None):
    """질문 문자열을 받아 {"answer": str, "sources": list[str]} 를 반환하는 체인을 만듭니다.

    반드시 지킬 것:
      - 검색된 문서 본문을 프롬프트에 실제로 넣어야 합니다.
      - sources는 실제 검색 결과의 metadata에서 뽑습니다. 지어내면 안 됩니다.

    심화 파트를 구현했다면 assess_retrieval 판정을 거쳐야 합니다.
    """
    llm = llm or _default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def answer_question(question: Any) -> dict:
        # 입력 정리: {"question": "..."} 형태로 들어와도, None이어도 문자열로 맞춥니다.
        if isinstance(question, dict):
            question = question.get("question", "")
        question = "" if question is None else str(question)

        # 빈 질문은 검색할 거리가 없으므로 검색·모델 호출 없이 바로 거절합니다.
        if not question.strip():
            return _no_answer()

        # 1) 검색
        docs = retriever.invoke(question) or []

        # 2) 검색 품질 판정(심화). 쓸 만하지 않으면 모델을 부르지 않고 거절합니다.
        verdict = assess_retrieval(docs, question)
        if not verdict["usable"]:
            return _no_answer()

        # 3) 검색된 본문을 프롬프트에 주입 -> 4) 모델 호출 -> 5) 텍스트만 꺼내기
        messages = prompt.format_messages(context=format_docs(docs), question=question)
        answer = get_text(llm.invoke(messages)).strip()

        # 모델이 스스로 "찾을 수 없다"고 답했다면 근거가 없는 것이므로 표준 거절 응답으로 통일합니다.
        if "찾을 수 없" in answer:
            return _no_answer()

        # 6) 출처는 실제 검색 결과의 metadata에서만 뽑습니다. (검색 순위 순서, 중복 제거)
        sources = list(dict.fromkeys(
            _file_name(d.metadata["source"])
            for d in docs
            if d.metadata.get("source")
        ))
        return {"answer": answer, "sources": sources}

    return RunnableLambda(answer_question)


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 질문 단어 끝에 붙는 조사·어미. 떼어 내야 '연차는'과 본문의 '연차'를 같은 말로 볼 수 있습니다.
_SUFFIXES = sorted(
    [
        "에서는", "으로는", "까지는", "부터는", "이라도", "되나요", "하나요", "있나요", "없나요",
        "인가요", "한가요", "합니다", "됩니다", "입니다",
        "에서", "으로", "에게", "까지", "부터", "보다", "처럼", "만큼", "이랑", "하고",
        "해도", "하면", "되면", "해야", "하는", "되는", "해요", "돼요", "나요", "까요", "세요",
        "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "와", "과", "로", "랑",
    ],
    key=len,
    reverse=True,  # 긴 것부터 비교해야 '에서'보다 '에서는'이 먼저 잘립니다.
)

# 어느 문서에나 나올 수 있어 판정에 도움이 안 되는 말(의문사·서술어·일반 명사)입니다.
_STOPWORDS = {
    "어떻게", "어떤", "무엇", "무엇인가요", "뭐", "뭐야", "뭔가요", "언제", "어디", "어디서",
    "누구", "얼마", "얼마나", "얼마인가요", "며칠", "몇일", "왜",
    "되나요", "하나요", "있나요", "없나요", "인가요", "한가요", "돼요", "해요", "있어", "있어요",
    "알려줘", "알려주세요", "궁금합니다", "가능", "가능한가요", "가능해요",
    "사내", "회사", "규정", "정책", "문서", "관련", "대해", "대한", "경우", "그거", "이거", "저거",
}


def _extract_keywords(question: str) -> list[str]:
    """질문에서 핵심어 목록을 뽑습니다. (중복 제거, 등장 순서 유지)"""
    # 한글·영문·숫자 덩어리를 단어로 봅니다. 'TR-102'처럼 하이픈으로 이어진 코드는 한 단어로 둡니다.
    tokens = re.findall(r"[0-9A-Za-z가-힣]+(?:-[0-9A-Za-z가-힣]+)*", question)
    keywords: list[str] = []
    for token in tokens:
        word = token.lower()
        if word in _STOPWORDS:
            continue
        for suffix in _SUFFIXES:
            # 떼고 남은 말이 2글자 이상일 때만 뗍니다. ('휴가'에서 '가'를 떼면 뜻이 사라지므로)
            if word.endswith(suffix) and len(word) - len(suffix) >= 2:
                word = word[: -len(suffix)]
                break
        if len(word) >= 2 and word not in _STOPWORDS and word not in keywords:
            keywords.append(word)
    return keywords


def assess_retrieval(docs, question: str) -> dict:
    """검색 결과가 이 질문에 답하기에 쓸만한지 판정합니다.

    이 함수는 LLM을 호출하지 않습니다. 같은 입력에 항상 같은 출력을 내야 합니다.

    판정 방법은 자유입니다. 예를 들면
      - 질문에서 핵심어를 뽑아 문서 본문에 몇 개나 등장하는지 센다
      - 검색된 문서 개수와 길이를 본다
      - 특정 임계값을 넘는지 확인한다

    Returns:
        {"usable": bool, "reason": str, "matched": int} 형태의 dict.
        usable이 False면 build_rag_chain은 모델을 부르지 않고 거절해야 합니다.
    """
    # 판정 규칙: "질문 핵심어의 절반 이상이 '한 문서' 본문 안에 함께 등장하면 쓸 만하다."
    #   - 여러 문서에 흩어져 하나씩 맞는 것은 우연일 가능성이 커서, 문서별 최댓값을 씁니다.
    #   - 본문(page_content)만 보고 메타데이터(제목)는 보지 않습니다. 제목은 같은 문서의
    #     모든 청크에 붙어 있어서 넣으면 무관한 청크까지 통과시키기 때문입니다.
    docs = list(docs or [])
    if not docs:
        return {"usable": False, "reason": "검색 결과가 없습니다.", "matched": 0}

    keywords = _extract_keywords(str(question or ""))
    if not keywords:
        return {"usable": False, "reason": "질문에서 핵심어를 찾지 못했습니다.", "matched": 0}

    texts = [str(getattr(d, "page_content", "") or "").lower() for d in docs]
    matched = max(sum(1 for kw in keywords if kw in text) for text in texts)
    required = (len(keywords) + 1) // 2  # 절반 이상(올림)

    usable = matched >= required
    reason = (
        f"질의어 {len(keywords)}개({', '.join(keywords)}) 중 {matched}개가 한 문서 본문에 등장 "
        f"(기준 {required}개 이상 -> {'통과' if usable else '거절'})"
    )
    return {"usable": usable, "reason": reason, "matched": matched}


NO_ANSWER = {
    "answer": "제공된 문서에서 답을 찾을 수 없습니다.",
    "sources": [],
}


def _no_answer() -> dict:
    """거절 응답의 복사본을 돌려줍니다. (원본 NO_ANSWER가 실수로 바뀌는 것을 막기 위함)"""
    return {"answer": NO_ANSWER["answer"], "sources": []}


# ══════════════════════════════════════════════════════════════════
# 직접 실행용 데모 — 저장된 Chroma DB로 실제 질문을 던져 봅니다
# (채점기는 이 아래를 실행하지 않습니다. `python submission.py` 로 직접 실행할 때만 동작)
# ══════════════════════════════════════════════════════════════════

def _load_vectorstore():
    """디스크에 저장된 Chroma DB를 불러옵니다. 새로 인덱싱하지 않으므로 임베딩 비용이 들지 않습니다.

    질문을 벡터로 바꿀 때 임베딩 모델이 필요하므로 embedding_function은 여전히 지정합니다.
    """
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=_default_embeddings(),
        persist_directory=PERSIST_DIR,
    )


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()  # .env 파일의 AWS 접속 정보를 환경변수로 읽어 옵니다.

    chunks = build_chunks([])  # 경로 목록이 비면 실습 docs 폴더 전체를 읽습니다.
    print(f"[청킹] {len(chunks)}개 청크 생성")
    for c in chunks[:3]:
        print(f"  {c.metadata} -> {c.page_content[:30].replace(chr(10), ' ')}...")

    vectorstore = _load_vectorstore()
    print(f"\n[Chroma] '{COLLECTION}' 컬렉션 청크 수: {vectorstore._collection.count()}")

    retriever = build_retriever(vectorstore)
    chain = build_rag_chain(retriever)

    questions = [
        "연차는 며칠까지 이월되나요?",            # fact
        "국내 출장 식비 한도는 하루 얼마인가요?",  # fact
        "재택근무는 주 며칠까지 가능한가요?",      # fact
        "사내 헬스장 운영 시간이 어떻게 되나요?",  # no_answer
    ]
    for q in questions:
        verdict = assess_retrieval(retriever.invoke(q), q)
        result = chain.invoke(q)
        print("\n" + "=" * 60)
        print(f"질문: {q}")
        print(f"판정: {verdict}")
        print(f"답변: {result['answer']}")
        print(f"출처: {result['sources']}")


if __name__ == "__main__":
    main()
