import os

from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI

from src.common.methods import set_separate_line

load_dotenv()


def conversation_with_context():
    set_separate_line("컨텍스트 기반 대화 예제")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)

    system_context = """
너는 프로그래밍 학습을 도와주는 AI 튜터야.
학습자의 수준을 파악하고 적절한 난이도로 설명해줘.
실제 코드 예제를 포함해서 답변해.
"""

    memory = ConversationBufferWindowMemory(k=6, return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_context),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    learning_conversation = [
        "안녕! 나는 이제 막 파이썬을 공부하기 시작했어.",
        "변수가 정확히 어떤거야?",
        "그럼 변수 이름을 지을 때 규칙이 있어?",
        "실제 예제를 보여줄래?",
    ]

    for user_input in learning_conversation:
        chat_history = memory.chat_memory.messages
        response = chain.invoke({"input": user_input, "chat_history": chat_history})

        print(f"\nUSER: {user_input}")
        print(f"AI: {response}")

        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 오류")
        return

    try:
        conversation_with_context()
    except Exception as e:
        print(f"실행 중 오류: {e}")


if __name__ == "__main__":
    main()
