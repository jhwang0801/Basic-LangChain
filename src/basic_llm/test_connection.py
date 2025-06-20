import os
import sys

from langchain_openai import ChatOpenAI

from config.settings import settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_openai_connection():
    try:
        settings.validate_api_keys()

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY,
        )

        response = llm.invoke("안녕! 간단하게 인사해줘.")
        print("응답:", response.content)

    except Exception as e:
        print(f"연결 실패: {e}")


if __name__ == "__main__":
    test_openai_connection()
