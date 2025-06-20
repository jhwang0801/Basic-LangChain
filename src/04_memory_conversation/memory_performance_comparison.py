comparison_data = {
    "ConversationBufferMemory": {
        "장점": "완전한 대화 기록 유지",
        "단점": "토큰 사용량 증가",
        "적용": "짧은 대화, 정확한 기록 필요",
    },
    "ConversationBufferWindowMemory": {
        "장점": "일정한 메모리 사용량",
        "단점": "오래된 정보 손실",
        "적용": "긴 대화, 메모리 제한",
    },
    "ConversationSummaryMemory": {
        "장점": "토큰 효율적, 핵심 정보 유지",
        "단점": "요약 과정에서 정보 손실 가능",
        "적용": "매우 긴 대화, 비용 민감",
    },
    "ConversationSummaryBufferMemory": {
        "장점": "최근 대화 + 요약의 장점",
        "단점": "복잡한 관리",
        "적용": "대부분의 실제 애플리케이션",
    },
}
