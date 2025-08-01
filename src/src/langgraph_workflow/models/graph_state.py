import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
# node 간에 전달되어야하는 instance나 data를 정의의
class AgentState(TypedDict):
    """
    워크플로우의 상태를 나타내는 TypedDict입니다.
    노드 간에 메시지 목록을 전달합니다.
    """



    # init_state
    user_input: str
    customer_reviews: str  # 감성 분석을 위한 고객 리뷰
    # 노드 간에 전달되어야하는 데이터
    messages: Annotated[List[BaseMessage], operator.add]
    # 노드 간에 전달되어야하는 데이터
    regression_return: str
    sentiment_return: dict  # 감성 분석 결과
    rag_return: str  # RAG 검색 및 요약 결과
    final_report: str # 최종 생성된 보고서
    project_root: str # 프로젝트 루트 경로 추가
    # 노드 간에 전달되어야하는 데이터
    model_return: str
    # 노드 간에 전달되어야하는 데이터
    tool_return: str
    # 노드 간에 전달되어야하는 데이터