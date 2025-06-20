import os
import sys

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config.settings import settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def example_1_basic_invoke():
    print("=== Invoke ===")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0.3, api_key=settings.OPENAI_API_KEY
    )

    response = llm.invoke("파이썬의 장점 3가지를 간단히 설명해주세요.")
    print(response.content)
    print()


def example_2_message_types():
    print("=== 메시지 타입 ===")

    llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY)

    messages = [
        SystemMessage(content="당신은 친근한 파이썬 튜터입니다."),
        HumanMessage(content="리스트 컴프리헨션을 설명해주세요."),
    ]

    response = llm.invoke(messages)
    print(response.content)
    print()


def example_3_parameters():
    print("=== 파라미터 조절 ===")

    # 창의적인 응답 (temperature=0.9)
    creative_llm = ChatOpenAI(
        temperature=0.9, max_tokens=100, api_key=settings.OPENAI_API_KEY
    )

    # 일관된 응답 (temperature=0.1)
    consistent_llm = ChatOpenAI(
        temperature=0.1, max_tokens=100, api_key=settings.OPENAI_API_KEY
    )

    question = "인공지능의 미래를 한 문장으로 표현해주세요."

    print("창의적 응답 (temp=0.9):")
    print(creative_llm.invoke(question).content)
    print()

    print("일관된 응답 (temp=0.1):")
    print(consistent_llm.invoke(question).content)
    print()


if __name__ == "__main__":
    try:
        settings.validate_api_keys()

        example_1_basic_invoke()
        example_2_message_types()
        example_3_parameters()

    except Exception as e:
        print(f"오류 발생: {e}")
