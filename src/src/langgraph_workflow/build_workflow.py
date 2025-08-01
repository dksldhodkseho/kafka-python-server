from langgraph.graph import StateGraph, END
from .nodes import (
    call_regression_model_node,
    call_sentiment_analysis_node,
    call_rag_node,
    generate_report_node
)
from .models.graph_state import AgentState

def build_workflow():
    """
    LangGraph 워크플로우를 단순한 순차 파이프라인으로 구축하고 컴파일합니다.
    """
    # 그래프 정의
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("regression", call_regression_model_node)
    workflow.add_node("sentiment_analysis", call_sentiment_analysis_node)
    workflow.add_node("rag", call_rag_node)
    workflow.add_node("generate_report", generate_report_node)

    # 엣지 추가 (순차적으로 연결)
    workflow.set_entry_point("regression")
    workflow.add_edge("regression", "sentiment_analysis")
    workflow.add_edge("sentiment_analysis", "rag")
    workflow.add_edge("rag", "generate_report")
    workflow.add_edge("generate_report", END)

    # 그래프 컴파일
    return workflow.compile()
