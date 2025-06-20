import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

load_dotenv()


def set_separate_line(description: str):
    print("=" * 60)
    print(description)
    print("=" * 60)


def basic_chat_example():
    set_separate_line("기본 채팅 예제")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=200)

    # 1. 간단한 질문
    print("1. 간단한 질문:")
    response = llm.invoke("파이썬과 자바스크립트의 차이점을 3줄로 설명해주세요.")
    print(f"   {response.content}\n")

    # 2. 메시지 리스트 사용
    print("2. 메시지 리스트 사용:")
    messages = [
        SystemMessage(content="당신은 친근한 프로그래밍 튜터입니다."),
        HumanMessage(content="LangChain이 뭔가요?"),
    ]
    response = llm.invoke(messages)
    print(f"   {response.content}\n")


def message_types_example():
    set_separate_line("메시지 타입별 예제")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=150)

    conversation = [
        SystemMessage(
            content="당신은 요리 전문가입니다. 간단하고 실용적인 조언을 해주세요."
        ),
        HumanMessage(content="알리올리오를 맛있게 만드는 비법이 있나요?"),
    ]

    response1 = llm.invoke(conversation)
    print("요리사: " + response1.content + "\n")

    # 대화에 AI 응답 추가
    conversation.append(AIMessage(content=response1.content))
    conversation.append(HumanMessage(content="그럼 소스는 어떻게 만들어야 하나요?"))

    # 두 번째 응답 (맥락 유지)
    response2 = llm.invoke(conversation)
    print("요리사: " + response2.content + "\n")


def temperature_comparison():
    set_separate_line("Temperature 설정 비교")

    question = "AI의 미래에 대해 한 문장으로 말해주세요."

    temperatures = [0.1, 0.5, 0.9]

    for temp in temperatures:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temp, max_tokens=100)

        response = llm.invoke(question)
        print(f"Temperature {temp}: {response.content}\n")


def token_usage_tracking():
    set_separate_line("토큰 사용량 추적")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    questions = [
        "안녕하세요",
        "파이썬 리스트와 튜플의 차이점을 설명해주세요",
        "머신러닝 알고리즘 중에서 회귀 분석, 분류, 클러스터링의 차이점과 각각의 실제 사용 사례를 자세히 설명해주세요",
    ]

    total_tokens = 0

    for i, question in enumerate(questions, 1):
        response = llm.invoke(question)

        # 대략적인 토큰 수 계산 >> 실무에서는 tiktoken 이라는 라이브러리 사용 (https://github.com/openai/tiktoken)
        input_tokens = len(question.split()) * 1.3
        output_tokens = len(response.content.split()) * 1.3
        question_total = input_tokens + output_tokens
        total_tokens += question_total

        print(f"{i}. 질문: {question[:30]}...")
        print(f"   응답: {response.content[:50]}...")
        print(f"   예상 토큰: ~{question_total:.0f} tokens\n")

    print(f"총 예상 토큰: ~{total_tokens:.0f} tokens")
    print(f"예상 비용: ~${total_tokens * 0.000002:.6f} (GPT-3.5-turbo 기준)")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY오류")
        return

    try:
        basic_chat_example()
        message_types_example()
        temperature_comparison()
        token_usage_tracking()

    except Exception as e:
        print(f"실행 중 오류: {e}")


if __name__ == "__main__":
    main()
