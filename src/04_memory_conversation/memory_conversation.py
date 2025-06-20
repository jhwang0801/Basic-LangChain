import os

from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from src.common.methods import set_separate_line

load_dotenv()


# ConversationBufferMemory (1.0.0 버전 > deprecated 예정)
def basic_memory_example():
    set_separate_line("기본 메모리 예제")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)

    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

    conversations = [
        "안녕! 나는 LangChain을 공부하고 있는 사람이야.",
        "LangChain과 LangGraph 차이는 뭐야?",
        "그럼 언제 LangChain을 쓰고 언제 LangGraph를 써야해?",
        "내가 뭘 공부하고 있다고 했지?",
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"user({i}): {user_input}")
        response = conversation.predict(input=user_input)
        print(f"AI({i}): {response}")

    print("\nChat History:")
    print(memory.chat_memory.messages)


# RunnableWithMessageHistory
def recent_memory_example():
    set_separate_line("제일 최근 메모리 예제")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "너는 매우 도움되는 어시스턴트야."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    conversations = [
        "안녕! 나는 LangChain을 공부하고 있는 사람이야.",
        "LangChain과 LangGraph 차이는 뭐야?",
        "그럼 언제 LangChain을 쓰고 언제 LangGraph를 써야해?",
        "내가 뭘 공부하고 있다고 했지?",
    ]

    session_id = "user_session_1"

    for i, user_input in enumerate(conversations, 1):
        print(f"user({i}): {user_input}")
        response = with_message_history.invoke(
            {"input": user_input}, config={"configurable": {"session_id": session_id}}
        )
        print(f"AI({i}): {response.content}")

    print("\nChat History:")
    history = get_session_history(session_id)
    for message in history.messages:
        print(f"{message.__class__.__name__}: {message.content}")


def window_memory_example():
    set_separate_line("윈도우 메모리 예제, 최근 N개의 대화만 기억")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

    memory = ConversationBufferWindowMemory(k=3)  # 최근 4개의 메세지만 기억
    conversation = ConversationChain(llm=llm, memory=memory)

    conversations = [
        "나는 김아무개야.",
        "나이는 20살이야.",
        "취미는 영화감상이야.",
        "개발자로 일하고 있어.",
        "내 이름이 뭐라고?",  # 잊혀져야함
        "내 직업이 뭐라고?",  # 제대로 나와야함
    ]

    for i, user_input in enumerate(conversations, 1):
        response = conversation.predict(input=user_input)
        print(f"\nUSER({i}): {user_input}")
        print(f"AI({i}): {response}")

    print("\nChat History:")
    print(memory.chat_memory.messages)


def summary_buffer_memory_example():
    set_separate_line("서머리 버퍼 메모리, 최근대화 + 이전 요약")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)

    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=200)

    conversation = ConversationChain(llm=llm, memory=memory)
    extended_conversation = [
        "안녕! 나는 김똑개이고 대학생이야.",
        "컴퓨터공학과 3학년이고, 현재 졸업 프로젝트로 웹 애플리케이션을 개발해.",
        "django와 react를 사용해서 서비스를 만들고 있어.",
        "데이터베이스는 Postgresql를 사용하고, AWS에 배포할거야.",
        "백엔드 개발이 재미있어서 앞으로 이 분야로 취업하고 싶어.",
        "내가 어떤 기술스택을 사용한다고 했는지 기억해?",
    ]

    for i, user_input in enumerate(extended_conversation, 1):
        response = conversation.predict(input=user_input)
        print(f"\nUSER({i}): {user_input}")
        print(f"AI({i}): {response}")

        if hasattr(memory, "moving_summary_buffer") and memory.moving_summary_buffer:
            print(f"SUMMARY: {memory.moving_summary_buffer}")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 오류")
        return

    try:
        # basic_memory_example()
        # recent_memory_example()
        # window_memory_example()
        summary_buffer_memory_example()

    except Exception as e:
        print(f"실행 중 오류: {e}")


if __name__ == "__main__":
    main()
