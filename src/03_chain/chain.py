import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.common.methods import set_separate_line

load_dotenv()


def simple_chain_example():
    set_separate_line("체인 간단 예제")
    prompt = ChatPromptTemplate.from_template(
        "다음 키워드를 이용하여 창의적인 이야기를 만들어줘: {keyword}"
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=100)

    # 체인 구성 (LangChain Expression Language(LCEL))
    chain = prompt | llm | StrOutputParser()

    keywords = ["우주", "괴물", "타임머신", "마법", "개발자"]

    for keyword in keywords:
        result = chain.invoke({"keyword": keyword})
        print(f"{keyword}: {result}\n")


def sequential_chain_example():
    set_separate_line("순차 체인 예제")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)

    # step 1. 주제 분석
    analysis_prompt = ChatPromptTemplate.from_template(
        "'{topic}' 주제의 핵심 키워드 3개를 콤마로 구분하여 나열해."
    )
    analysis_chain = analysis_prompt | llm | StrOutputParser()

    # step 2. 키워드 기반 내용 생성
    content_prompt = ChatPromptTemplate.from_template(
        "다음 키워드들을 활용하여 블로그 글 제목을 만들어줘: {keywords}"
    )
    content_chain = content_prompt | llm | StrOutputParser()

    topics = ["인공지능과 일자리", "메타버스와 교육", "이란과 이스라엘 간의 전쟁"]

    for topic in topics:
        print(f"topic: {topic}")

        # step 1
        keywords = analysis_chain.invoke({"topic": topic})
        print(f"STEP 1 키워드 결과: {keywords}")

        # step 2
        content = content_chain.invoke({"keywords": keywords})
        print(f"STEP 2 컨텐츠 결과: {content}\n")


def conditional_chain_example():
    set_separate_line("조건부 체인 예제")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=100)

    def route_question(question):
        tech_keywords = ["파이썬", "프로그래밍", "알고리즘"]

        if any(keyword in question.lower() for keyword in tech_keywords):
            return "tech"
        else:
            return "general"

    tech_prompt = ChatPromptTemplate.from_template(
        "기술 전무가로서 다음 질문에 정확하고 상세히 답변해줘: {question}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "치근하고 이해하기 쉽게 다음 질문에 답변해줘: {question}"
    )

    questions = [
        "기분이 좋아지는 방법은?",
        "파이썬 딕셔너리를 사용하는 방법은?",
        "여행 갈 때 챙겨야할 가장 중요한 준비물은?",
    ]

    for question in questions:
        question_type = route_question(question)

        if question_type == "tech":
            final_prompt = tech_prompt
        else:
            final_prompt = general_prompt

        chain = final_prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question})
        print(f"Q: [{question_type}] {question}")
        print(f"A: {response}\n")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 오류")
        return

    try:
        simple_chain_example()
        sequential_chain_example()
        conditional_chain_example()

    except Exception as e:
        print(f"실행 중 오류: {e}")


if __name__ == "__main__":
    main()
