import os

from dotenv import load_dotenv
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

from src.common.methods import set_separate_line

load_dotenv()


def basic_prompt_template():
    """
    PromptTemplate(
                template=template,
                input_variables=input_variables,
                partial_variables=partial_variables,
                template_format="f-string",
                validate_template=validate_template,
            )
    """
    set_separate_line("기본 프롬프트 템플릿 테스트")

    template = """
    주제: {topic}
    난이도: {level}
    언어: 한국어
    
    위 조건에 맞추어 {topic}에 대해 {level} 수준으로 설명해줘.
    """

    prompt = PromptTemplate(template=template, input_variables=["topic", "level"])
    topics = [
        {"topic": "머신러닝", "level": "초급자"},
        {"topic": "양자역학", "level": "중급자"},
        {"topic": "양자컴퓨팅", "level": "전문가"},
    ]

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=200)

    for data in topics:
        formatted_prompt = prompt.format(**data)
        response = llm.invoke(formatted_prompt)

        print(f"formatted_prompt: {formatted_prompt}\n")
        print(f"raw_response: {response}\n")
        print(f"{data['topic']}({data['level']}): {response.content}\n")


def chat_prompt_template():
    set_separate_line("채팅 프롬프트 템플릿")
    system_template = "너는 {role}이야. {style}로 답변해"
    human_template = "{question}"

    # chat_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_template),
    #         ("human", human_template),
    #     ]
    # )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8, max_tokens=150)

    scenarios = [
        {
            "role": "친근한 프로그래밍 튜터",
            "style": "쉽고 재미있게",
            "question": "파이썬 리스트 컴프리헨션이 뭔가요?",
        },
        {
            "role": "엄격한 대학 교수",
            "style": "학술적이고 정확하게",
            "question": "알고리즘 복잡도란 무엇인가?",
        },
        {
            "role": "유머러스한 개발자",
            "style": "농담을 섞어서",
            "question": "버그가 왜 생기나요?",
        },
    ]

    for scenario in scenarios:
        messages = chat_prompt.format_messages(**scenario)
        response = llm.invoke(messages)

        print(f"답변: {response.content}\n")


def partial_prompt_template():
    set_separate_line("파셜 템플릿 활용")

    base_template = PromptTemplate(
        template="""
회사: {company}
포지션: {position}
경력: {experience}년
기술스택: {skills}

위 정보를 바탕으로 {task}를 작성해줘.
형식: {format}
길이: {length}
""",
        input_variables=[
            "company",
            "position",
            "experience",
            "skills",
            "task",
            "format",
            "length",
        ],
    )

    # 고정값(?)
    partial_prompt = base_template.partial(
        company="테크스타트업", format="불렛포인트", length="5줄 이내"
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, max_tokens=200)

    # 변동값(?)
    job_profiles = [
        {
            "position": "백엔드 개발자",
            "experience": 3,
            "skills": "Python, Django, PostgreSQL",
            "task": "자기소개서",
        },
        {
            "position": "프론트엔드 개발자",
            "experience": 2,
            "skills": "React, TypeScript, Next.js",
            "task": "기술 블로그 소개글",
        },
    ]

    for profile in job_profiles:
        formatted_prompt = partial_prompt.format(**profile)
        response = llm.invoke(formatted_prompt)

        print(f"{profile['position']} ({profile['experience']}년차)")
        print(f"{response.content}\n")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 오류")
        return

    try:
        basic_prompt_template()
        chat_prompt_template()
        partial_prompt_template()

    except Exception as e:
        print(f"실행 중 오류: {e}")


if __name__ == "__main__":
    main()
