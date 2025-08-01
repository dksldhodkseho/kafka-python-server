from typing import TypedDict, List
import json
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field

from langgraph_workflow.models.graph_state import AgentState
from regression_model.regression import RegressionModel
from sentiment_analysis.analysis import SentimentAnalyzer
from rag_retriever.retriever import RagRetriever

def call_regression_model_node(state: AgentState): # 타입을 AgentState로 변경
    """RegressionModel을 호출하고 결과를 JSON 문자열로 상태에 추가합니다."""
    print("---회귀 모델 호출---")
    user_input = state["user_input"]
    # RegressionModel은 한 번만 초기화되어야 효율적이지만, 여기서는 단순성을 위해 매번 생성합니다.
    regression_model = RegressionModel()
    prediction_json = regression_model.predict(user_input)
    return {"regression_return": prediction_json}

def call_sentiment_analysis_node(state: AgentState): # 타입을 AgentState로 변경
    """SentimentAnalyzer를 호출하여 고객 리뷰를 분석하고 결과를 상태에 추가합니다."""
    print("---고객 리뷰 감성 분석 호출---")
    reviews = state["customer_reviews"]
    analyzer = SentimentAnalyzer()
    analysis_result = analyzer.analyze(reviews)
    return {"sentiment_return": analysis_result}

def call_rag_node(state: AgentState): # 타입을 AgentState로 변경
    """
    RAG Retriever를 호출하여 시장 및 트렌드 분석을 수행합니다.
    """
    print("---시장 동향 분석(RAG) 호출---")
    user_input = state["user_input"]
    retriever = RagRetriever()
    rag_return = retriever.retrieve_and_summarize(user_input)
    return {"rag_return": rag_return}

def generate_report_node(state: AgentState): # 타입을 AgentState로 변경
    """
    지금까지 수집된 모든 데이터를 바탕으로 최종 보고서를 생성합니다.
    """
    print("---최종 보고서 생성 시작---")
    
    # 1. 회귀 분석 결과(JSON) 파싱 및 자연스러운 문장으로 변환
    try:
        reg_data = json.loads(state['regression_return'])
        sales_prediction = reg_data.get('sales_prediction', 'N/A')
        prediction_name = reg_data.get('name', 'N/A')
        
        reg_type = reg_data.get('type')
        if reg_type == 'product_specific':
            reg_text = f"제품 '{prediction_name}'의 고유 데이터를 기반으로 예측한 다음 달 예상 매출액은 {sales_prediction} 입니다."
        elif reg_type == 'sub_category_fallback':
            # "데이터 부족"이라는 직접적인 언급 대신, 카테고리 예측임을 명시
            reg_text = f"제품 카테고리 '{prediction_name}'의 데이터를 기반으로 예측한 다음 달 예상 매출액은 {sales_prediction} 입니다."
        elif reg_type == 'sub_category_specific':
            reg_text = f"카테고리 '{prediction_name}' 전체의 데이터를 기반으로 예측한 다음 달 예상 매출액은 {sales_prediction} 입니다."
        else: # error case
            reg_text = f"예측 실패: {reg_data.get('reason', '알 수 없는 오류')}"

        # 컨텍스트 정보가 있는 경우, 평가 문구 추가
        if 'context' in reg_data:
            avg_sales = reg_data['context'].get('avg_sales', 'N/A')
            max_sales = reg_data['context'].get('max_sales', 'N/A')
            reg_text += (
                f" 이 예측치는 해당 카테고리(또는 상품)의 과거 평균 월 매출({avg_sales}) 및 "
                f"최고 월 매출({max_sales}) 데이터와 비교하여 평가되었습니다."
            )

        if reg_data.get('type') == 'sub_category_fallback':
            reason = reg_data.get('reason', '데이터 부족으로 상위 카테고리 모델 사용')
            reg_text = f"제품 '{prediction_name}'에 대한 예측: {reg_text} ({reason})"

    except (json.JSONDecodeError, TypeError):
        reg_text = f"회귀 분석 결과(문자열): {state['regression_return']}"

    # 2. 감성 분석 결과(dict)를 텍스트로 변환
    # 키 이름을 'sentiment_analysis_result'에서 'sentiment_return'으로 수정
    sentiment_text = json.dumps(state.get("sentiment_return", {}), indent=2, ensure_ascii=False)
    
    # 3. RAG 결과(str)
    rag_text = state.get("rag_return", "데이터 없음")

    # 4. LLM에 전달할 프롬프트 정의
    prompt_template = f"""
당신은 냉철한 비즈니스 분석가입니다. 주어진 데이터를 바탕으로 특정 상품의 현황을 분석하고, 향후 운영 방향을 결정하여 상세한 보고서를 작성해야 합니다.

아래의 3가지 데이터를 사용하여 종합적으로 판단하고, 최종 보고서를 "최종 결론", "종합 평가", "세부 분석 및 판단 근거", "최종 권고 사항"의 4가지 항목으로 나누어 작성해주세요.

---
[데이터 1: 정량 예측 데이터 (미래 판매량)]
{reg_text}

[데이터 2: 고객 반응 데이터 (리뷰 기반 감성 분석)]
{sentiment_text}

[데이터 3: 시장 환경 데이터 (RAG 기반 시장 조사)]
{rag_text}
---

[보고서 작성 가이드라인]
- **세부 분석 및 판단 근거**: 각 데이터가 의미하는 바를 심층적으로 분석하고, 데이터 간의 연관성을 설명해주세요.
  - **정량 분석**: 예측된 판매량이 과거 평균/최대치와 비교하여 어떤 의미를 가지는지 구체적으로 해석하고, 이 수치가 긍정적인지, 부정적인 신호인지 명확히 판단해주세요.
  - **고객 반응 심층 분석**: 고객 리뷰에서 나타난 긍정적, 부정적 피드백의 **핵심 주제(예: 품질, 배송, 조립, 디자인)를 명확히 분류**하고, **각 주제에 대한 구체적인 리뷰 내용을 1~2개씩 인용**하여 주장을 뒷받침해주세요. 이 피드백이 판매량 예측에 어떤 영향을 미칠 수 있는지 연결하여 설명해주세요.
  - **시장 환경 분석**: (데이터가 있는 경우) 시장 트렌드가 현재 제품에 유리한지 불리한지 분석하고, 경쟁 상황을 고려하여 종합적인 판단을 내려주세요.
- **종합 판단**: 개별 분석들을 종합하여, 이 상품이 현재 시장에서 어떤 상태인지 (성장 가능성, 유지 필요, 개선 시급, 위험 등) 명확하게 평가해주세요.
- **최종 권고**: 종합 판단을 바탕으로, 앞으로 이 상품을 어떻게 운영해야 할지에 대한 구체적인 권고 사항을 제시해주세요. (예: 제품 개선, 마케팅 강화, 가격 조정 등)
- **단종 고려 조건**: 만약 모든 데이터를 종합적으로 분석했을 때, **판매량 예측치가 과거 평균에 비해 현저히 낮고, 고객 리뷰에서 심각한 결함이 지속적으로 언급되며, 시장 트렌드도 비관적이라면** '제품 단종 고려'를 포함한 과감한 권고를 내릴 수 있습니다. 그 외의 경우에는 구체적인 개선안을 제시해주세요.

결과는 반드시 한국어로, 위에 제시된 4가지 항목을 모두 포함한 보고서 형식으로만 출력해주세요.
"""
    
    # 5. LLM 모델 초기화 및 체인 실행
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    # f-string으로 완성된 프롬프트를 사용하므로, ChatPromptTemplate.from_template을 직접 사용하지 않습니다.
    # 대신, 완성된 문자열을 HumanMessage로 감싸서 전달합니다.
    chain = llm
    
    response = chain.invoke(prompt_template)
    
    print("---최종 보고서 생성 완료---")
    return {"final_report": response.content}
