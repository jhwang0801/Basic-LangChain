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


def custom_memory_example():
    set_separate_line("사용자 정보를 별도로 관리하는 커스텀 메모리 예제")

    class UserInfoMemory:
        def __init__(self):
            self.user_info = {}
            self.conversation_history = []

        def add_user_info(self, key, value):
            self.user_info[key] = value

        def add_conversation(self, user_msg, ai_msg):
            self.conversation_history.append({"user": user_msg, "ai": ai_msg})

            # 대화기록 5개까지만 유지
            if len(self.conversation_history) > 5:
                self.conversation_history.pop(0)

        def get_context(self):
            context = "사용자 정보:\n"
            for key, value in self.user_info.items():
                context += f"- {key}: {value}\n"

            context += "\n최근 대화:\n"
            for conversation in self.conversation_history[-3:]:
                context += f"사용자: {conversation['user']}\n"
                context += f"AI: {conversation['ai']}\n"

            return context

    custom_memory = UserInfoMemory()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, max_tokens=100)

    def chat_with_custom_memory(user_input):
        if "이름은" in user_input:
            name = user_input.split("이름은")[1].split()[0]
            custom_memory.add_user_info("이름", name)
        elif "나이" in user_input and "살" in user_input:
            age = [
                word for word in user_input.split() if word.replace("살", "").isdigit()
            ]
            if age:
                custom_memory.add_user_info("나이", age[0])
        elif "직업" in user_input or "일" in user_input:
            custom_memory.add_user_info("직업_관련", user_input)

        context = custom_memory.get_context()
        prompt = (
            f"{context}\n\n현재 질문: {user_input}\n\n위 정보를 고려해서 답변해주세요:"
        )

        response = llm.invoke(prompt)
        ai_response = response.content
        custom_memory.add_conversation(user_input, ai_response)

        return ai_response

    custom_conversations = [
        "안녕하세요! 제 이름은 박민수입니다.",
        "저는 28살이고 소프트웨어 개발자로 일하고 있어요.",
        "파이썬과 데이터 분석에 관심이 많습니다.",
        "제가 몇 살이라고 했죠?",
        "제 직업이 뭐라고 했나요?",
    ]

    for i, user_input in enumerate(custom_conversations, 1):
        ai_response = chat_with_custom_memory(user_input)
        print(f"\n사용자: {user_input}")
        print(f"AI: {ai_response}")

        if i == 3:  # 중간에 메모리 상태 출력
            print(f"\n저장된 사용자 정보: {custom_memory.user_info}")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 오류")
        return

    try:
        custom_memory_example()
    except Exception as e:
        print(f"실행 중 오류: {e}")


if __name__ == "__main__":
    main()
