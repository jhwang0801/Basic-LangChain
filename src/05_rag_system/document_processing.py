import os
import shutil
import tempfile

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.methods import set_separate_line

load_dotenv()


def create_sample_documents():
    set_separate_line("샘플 문서 생성")

    documents = [
        {
            "title": "똑똑한개발자 회사 소개",
            "content": """
            똑똑한개발자(TOKTOKHAN.DEV)는 디지털 프로덕트 에이전시로 웹/애플리케이션의 UX기획부터 디자인, 개발까지 프로덕트 전반에 걸쳐 서비스 경험을 만들어가고 있습니다.
            팀원 모두가 직무에 관계없이, 하나의 ‘프로덕트 개발자 (DEV)’라는 생각 아래, 역할을 한정짓지 않고 서로 배우며 성장합니다. **Plus, Connect, Think** 라는 슬로건은 우리의 이러한 방향성을 가장 잘 보여주고 있죠.
            우리는 에이전시지만 우리가 하는 일이 단순한 클라이언트 잡이 아닌, **팀의 성장**과 연결되어 있다고 믿어요. 
            **프로젝트별 회고와 피드백, 블로그 작성과 내부 스터디**를 통해 개인의 성장점을 서로 배우며 공유하고, 이를 바탕으로 더 큰 시너지를 내기 위한 지점을 고민하며 **채용문화(커리어스쿼드), 교육(인코스), 내부서비스 구축(SaaS), 컨퍼런스** 등의 고민과 노력을 이어가고 있습니다.
            회사의 CEO는 서장원이며, CTO는 윤준구, COO는 이지민입니다. 
            """,
        },
        {
            "title": "똑똑한개발자 개발스택",
            "content": """
            똑똑한개발자에서는 백엔드는 Django, 프론트엔드는 React, ReactNative를 사용하여 개발을 진행한다.
            클라우드 서비스로는 AWS를 사용하여 구성한다.
            """,
        },
    ]

    doc_objects = []
    for doc in documents:
        doc_obj = Document(
            page_content=doc["content"],
            metadata={"title": doc["title"], "source": f"{doc['title']}.txt"},
        )
        doc_objects.append(doc_obj)

    print(f"{len(doc_objects)}개의 샘플 문서 생성 완료")
    return doc_objects


def text_splitting_example(documents):
    set_separate_line("텍스트 분할 예제")

    splitters = {
        "small": RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
        ),
        "medium": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100),
        "big": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
    }

    for name, splitter in splitters.items():
        chunks = splitter.split_documents(documents)
        print(f"\n {name} 분할 결과:")
        print(f"   총 청크 수: {len(chunks)}")
        print(f"   첫 번째 청크 길이: {len(chunks[0].page_content)}자")
        print(f"   첫 번째 청크 내용: {chunks[0].page_content[:100]}...")

    # 최적으로 분할 설정
    optimal_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
    )

    final_chunks = optimal_splitter.split_documents(documents)
    print(f"\n최종 선택된 분할 방식: medium 청크 ({len(final_chunks)}개)")

    return final_chunks


def create_vector_store(chunks):
    set_separate_line("벡터 스토어 생성 (임베딩)")

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        temp_dir = tempfile.mkdtemp()

        print("텍스트 >>>> 벡터 변환 중...")
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=temp_dir
        )

        print(f"벡터 스토어 생성 완료")
        print(f"   저장된 문서 수: {len(chunks)}")
        print(f"   저장 위치: {temp_dir}")

        return vectorstore, temp_dir

    except Exception as e:
        print(f"벡터 스토어 생성 실패: {e}")
        return None, None


def similarity_search_example(vectorstore):
    set_separate_line("유사도 검색 테스트")

    if not vectorstore:
        print("벡터 스토어 없음")
        return

    search_queries = [
        "똑똑한개발자는 뭐하는 회사야?",
        "기술 스택은 어떤걸 쓰는 회사야?",
        "여기에 일을 맡기면 예상되는 장점이 뭐야?",
        "CEO는 누구야?",
        "CTO는 누구야?",
        "COO는 누구야?",
    ]

    for query in search_queries:
        print(f"\n 검색어: '{query}'")

        # 유사도 기반으로 상위 2개 결과
        results = vectorstore.similarity_search(query, k=2)

        for i, result in enumerate(results, 1):
            print(f"   결과 {i}: {result.metadata.get('title', 'Unknown')}")
            print(f"      내용: {result.page_content[:100]}...")

        # 점수와 함께 검색
        scored_results = vectorstore.similarity_search_with_score(query, k=1)
        if scored_results:
            doc, score = scored_results[0]
            print(f"   유사도 점수: {score:.4f} (낮을수록 유사)")


def create_rag_chain(vectorstore):
    set_separate_line("RAG 체인 생성")

    if not vectorstore:
        print("벡터 스토어 없음")
        return None

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=300)
    system_prompt = (
        "다음 문서들을 참고해서 질문에 답변해주세요.\n\n"
        "답변 가이드라인:\n"
        "1. 참고 문서의 내용을 기반으로 답변하세요\n"
        "2. 문서에 없는 내용은 '문서에 해당 정보가 없습니다'라고 명시하세요\n"
        "3. 간결하고 정확하게 답변하세요\n"
        "4. 관련된 추가 정보가 있다면 함께 제공하세요\n\n"
        "참고 문서: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # retriever 생성
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # document chain 생성
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # rag chain 생성
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\nRAG CHAIN 생성완료")
    return rag_chain


def rag_qa_example(rag_chain):
    set_separate_line("RAG 기반 질문답변 예제")

    if not rag_chain:
        print("NO RAG CHAIN")
        return

    questions = [
        "똑똑한개발자는 뭐하는 회사야?",
        "기술 스택은 어떤걸 쓰는 회사야?",
        "여기에 일을 맡기면 예상되는 장점이 뭐야?",
        "CEO는 누구야?",
        "CTO는 누구야?",
        "COO는 누구야?",
        "똑똑한개발자 CTO 윤준구님이 소유한 차량의 번호판 번호는 뭐야?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQ {i}: {question}")

        try:
            # RAG 실행
            result = rag_chain.invoke({"input": question})
            answer = result.get("answer", "답변 불가")
            context_docs = result.get("context", [])

            print(f"A: {answer}")

            # 참조한 문서
            if context_docs:
                print("참조 문서:")
                for j, doc in enumerate(context_docs, 1):
                    title = doc.metadata.get("title", "Unknown")
                    print(f"   {j}.  {title}")

        except Exception as e:
            print(f"ERROR: {e}")


def advanced_retrieval_example(vectorstore):
    set_separate_line("고급 검색 예제")

    if not vectorstore:
        print("NO VECTORSTORE")
        return

    query = "똑똑한개발자 C 레벨 구성원"

    print(f"검색어: '{query}'")

    # 1. 유사도 검색
    print("\n기본 유사도 검색 (k=2):")
    basic_results = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(basic_results, 1):
        print(f"   {i}. {doc.metadata.get('title')} - {doc.page_content[:80]}...")

    # 2. MMR (Maximum Marginal Relevance) 검색
    print("\nMMR 검색:")
    try:
        mmr_results = vectorstore.max_marginal_relevance_search(
            query,
            k=2,
            fetch_k=4,  # 후보 4개 중에 2개 선택
            lambda_mult=0.7,  # 관련성 vs 다양성
        )
        for i, doc in enumerate(mmr_results, 1):
            print(f"   {i}. {doc.metadata.get('title')} - {doc.page_content[:80]}...")

    except:
        print("   MMR 검색 지원 X")

    # 3. 메타데이터 필터링
    print("\n메타데이터 필터링:")
    try:
        filtered_results = vectorstore.similarity_search(
            query,
            k=2,
            filter={
                "title": "똑똑한개발자 개발스택"
            },  # 메타데이터 필터 (아예 동일한 값이 아닌 경우, 필터 X)
        )
        for i, doc in enumerate(filtered_results, 1):
            print(f"   {i}. {doc.metadata.get('title')} - {doc.page_content[:80]}...")
    except:
        print("   메타데이터 필터링은 지원되지 않는 설정입니다.")


def cleanup_resources(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"임시 디렉토리 정리 완료: {temp_dir}")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 오류")
        return

    try:
        # 1. 샘플 문서 생성
        documents = create_sample_documents()

        # 2. 텍스트 분할
        chunks = text_splitting_example(documents)

        # 3. 벡터 스토어 생성
        vectorstore, temp_dir = create_vector_store(chunks)

        if vectorstore:
            # 4. 유사도 검색 테스트
            # similarity_search_example(vectorstore)

            # 5. RAG 체인 생성 (새로운 방식)
            # rag_chain = create_rag_chain(vectorstore)

            # 6. RAG 질문답변
            # rag_qa_example(rag_chain)

            # 7. 고급 검색 기법
            advanced_retrieval_example(vectorstore)

    except Exception as e:
        print(f"실행 중 오류: {e}")

    finally:
        cleanup_resources(temp_dir)


if __name__ == "__main__":
    main()
