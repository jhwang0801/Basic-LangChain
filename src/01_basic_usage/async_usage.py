import asyncio
import time
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.common.methods import set_separate_line

load_dotenv()


async def basic_async_example():
    set_separate_line("기본 비동기 처리")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

    questions = ["파이썬이란?", "자바스크립트란?", "Go언어란?"]

    print("** 비동기로 3개 질문 동시 처리 중...")
    start_time = time.time()

    tasks = [llm.ainvoke(question) for question in questions]
    responses = await asyncio.gather(*tasks)

    async_time = time.time() - start_time

    for i, (question, response) in enumerate(zip(questions, responses), 1):
        print(f"\n질문 {i}: {question}")
        print(f"답변: {response.content}")

    print(f"\n비동기 처리 시간: {async_time:.2f}초")
    return async_time


async def sync_vs_async_comparison():
    set_separate_line("동기 vs 비동기 성능 비교")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=50)

    questions = [
        "1+1은?",
        "Python의 특징은?",
        "AI란 무엇인가요?",
        "웹 개발이란?",
        "데이터베이스란?",
    ]

    # 동기 처리
    print("\n동기 처리:")
    sync_start = time.time()

    sync_responses = []
    for i, question in enumerate(questions, 1):
        print(f"   처리 중... {i}/{len(questions)}")
        response = await llm.ainvoke(question)
        sync_responses.append(response)

    sync_time = time.time() - sync_start
    print(f"   완료! 소요 시간: {sync_time:.2f}초")

    # 비동기 처리
    print("\n비동기 처리:")
    async_start = time.time()

    tasks = [llm.ainvoke(question) for question in questions]
    async_responses = await asyncio.gather(*tasks)

    async_time = time.time() - async_start
    print(f"   완료! 소요 시간: {async_time:.2f}초")

    # 성능 비교
    speedup = sync_time / async_time if async_time > 0 else 0
    print(f"\n결과:")
    print(f"   속도 향상: {speedup:.1f}배")
    print(f"   시간 절약: {sync_time - async_time:.2f}초")

    return sync_time, async_time


async def batch_processing_example():
    """배치 처리 예제"""
    set_separate_line("배치 처리 예제")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=30)

    tasks_data = [f"{i} + {i + 1} = ?" for i in range(1, 11)]

    batch_size = 5  # 동시처리 작업 갯수

    start_time = time.time()
    all_results = []

    for i in range(0, len(tasks_data), batch_size):
        batch = tasks_data[i : i + batch_size]
        print(f"   배치 {i // batch_size + 1} 처리 중... ({len(batch)}개)")

        batch_tasks = [llm.ainvoke(task) for task in batch]
        batch_results = await asyncio.gather(*batch_tasks)

        all_results.extend(batch_results)

        await asyncio.sleep(0.1)

    total_time = time.time() - start_time

    print(f"\n총 시간: {total_time:.2f}초")
    print(f"결과:")
    for i, (question, result) in enumerate(zip(tasks_data, all_results), 1):
        print(f"   {question} >> {result.content.strip()}")


def sync_wrapper():
    asyncio.run(basic_async_example())
    asyncio.run(sync_vs_async_comparison())
    asyncio.run(batch_processing_example())


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY오류")
        return

    try:
        sync_wrapper()  # 비동기 함수들 실행

    except Exception as e:
        print(f"실행 중 오류: {e}")


if __name__ == "__main__":
    main()
