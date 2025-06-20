# Basic LangChain Study

LangChain 기초부터 RAG까지 개인 학습 기록

## 📚 학습 내용

- **LangChain 기초**: 체인, 프롬프트 템플릿, 메모리 관리
- **RAG 구현**: 문서 처리, 벡터 스토어, 유사도 검색
- **비동기 처리**: 성능 최적화 및 배치 처리
- **다양한 메모리**: Buffer, Window, Summary 메모리 비교

## 🛠 Tech Stack

- **LangChain** + **OpenAI API**
- **Chroma** (Vector DB)
- **Python**

## 📁 파일 구성

```
src/
├── basic_usage.py              # LangChain 기본 사용법
├── prompt_templates.py         # 프롬프트 템플릿 활용
├── chain.py                    # 체인 구성 및 연결
├── async_usage.py              # 비동기 처리 성능 비교
├── memory_conversation.py      # 다양한 메모리 타입
├── custom_memory.py            # 커스텀 메모리 구현
├── conversation_with_context.py # 컨텍스트 기반 대화
├── document_processing.py      # RAG 구현 (문서 처리 + 벡터 검색)
└── common/methods.py           # 공통 유틸리티
```

## 🚀 주요 예제

### 기본 사용법
- ChatOpenAI 모델 사용
- Temperature 설정 비교
- 토큰 사용량 추적

### 프롬프트 & 체인
- 동적 프롬프트 템플릿
- 순차/조건부 체인 구성
- LCEL(LangChain Expression Language) 활용

### 메모리 관리
- ConversationBufferMemory
- ConversationBufferWindowMemory  
- ConversationSummaryBufferMemory
- 커스텀 메모리 구현

### RAG 시스템
- 문서 분할 및 청킹
- 벡터 임베딩 생성
- 유사도 검색 최적화
- 질문-답변 체인 구축

### 성능 최적화
- 동기 vs 비동기 처리 비교
- 배치 처리 구현


```env
OPENAI_API_KEY=your_key_here
```

---

**개인 학습용 레포지토리** | 2025.06