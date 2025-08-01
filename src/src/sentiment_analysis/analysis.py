import requests
import time
import openai
import os
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# --- LLM이 반환할 데이터 구조 정의 ---
class TopicOpinion(BaseModel):
    """분석된 주제 또는 속성과 그에 대한 의견"""
    topic: str = Field(description="분석된 주제 또는 속성 (예: 배터리, 디자인, 가격)")
    opinion: str = Field(description="해당 주제에 대한 고객들의 의견 요약")

class DetailedAnalysis(BaseModel):
    """고객 리뷰에 대한 상세 분석 결과"""
    summary: str = Field(description="전체 리뷰 내용에 대한 한두 문장의 핵심 요약")
    points_for_improvement: List[TopicOpinion] = Field(description="고객 리뷰에서 드러난 구체적인 개선점 목록")
    key_strengths: List[TopicOpinion] = Field(description="고객 리뷰에서 드러난 구체적인 핵심 강점 목록")


class SentimentAnalyzer:
    def __init__(self):
        """
        허깅페이스 모델과 OpenAI LLM을 초기화합니다.
        """
        # 1. (선택적) 허깅페이스 분류기 초기화
        hf_model_name = "snunlp/KR-FinBert-SC"
        self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)
        self.sentiment_classifier = pipeline("sentiment-analysis", model=self.hf_model, tokenizer=self.hf_tokenizer)
        
        # 2. OpenAI LLM 초기화 (구조화된 출력 사용)
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
        self.structured_llm = self.llm.with_structured_output(DetailedAnalysis)

        # 3. LLM에 사용할 프롬프트 템플릿 정의
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 고객 리뷰를 분석하여 핵심 강점과 개선점을 정확하게 추출하는 전문 분석가입니다. 주어진 리뷰들을 기반으로, 반드시 세 가지 정보를 추출해야 합니다: 전체 내용을 아우르는 한두 문장의 요약, 구체적인 강점 목록, 그리고 구체적인 개선점 목록."),
                ("human", "다음 고객 리뷰들을 분석해주세요:\n\n{reviews}")
            ]
        )
        self.chain = self.prompt | self.structured_llm


    def analyze(self, customer_reviews: str):
        """
        고객 리뷰를 분석하여 전반적인 감성 점수와 상세 분석 결과를 반환합니다.
        """
        print(f"SentimentAnalyzer.analyze() called with reviews: {customer_reviews[:30]}...")

        # 1단계: (참고용) 허깅페이스 모델로 전체 긍/부정 점수 계산
        hf_results = self.sentiment_classifier(customer_reviews[:512])
        overall_sentiment = {}
        for result in hf_results:
            label = result['label'].lower()
            overall_sentiment[label] = result['score']

        # 2단계: LLM으로 요약, 강점, 개선점 추출
        print("LLM을 호출하여 상세 분석을 시작합니다...")
        detailed_result = self.chain.invoke({"reviews": customer_reviews})
        
        # 3단계: 두 분석 결과 통합
        return {
            'overall_sentiment': overall_sentiment,
            'summary': detailed_result.summary,
            'points_for_improvement': [item.dict() for item in detailed_result.points_for_improvement],
            'key_strengths': [item.dict() for item in detailed_result.key_strengths]
        }