import sys
from pathlib import Path
import os
from typing import List

# --- 경로 설정 ---
# 이 main.py 파일의 위치를 기준으로 AI/src 폴더의 절대 경로를 계산합니다.
# main.py -> langgraph_workflow -> src. 이므로 두 단계 상위 폴더가 src 폴더입니다.
SRC_ROOT = Path(__file__).resolve().parents[1]
# 파이썬의 모듈 검색 경로에 AI/src 폴더를 추가합니다.
# 이렇게 하면 'from langgraph_workflow.build_workflow ...' 처럼 src 내부의 모든 모듈을 절대 경로로 참조할 수 있습니다.
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# .env 파일 로드 (AI 폴더에 있는 .env 파일을 명시적으로 지정)
try:
    from dotenv import load_dotenv
    dotenv_path = SRC_ROOT.parent / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"✅ .env 파일 로드 성공: {dotenv_path}")
    else:
        print("💡 .env 파일을 찾을 수 없습니다. API 키가 환경 변수에 설정되어 있어야 합니다.")
except ImportError:
    print("💡 python-dotenv가 설치되지 않았습니다. .env 파일을 사용하지 않습니다.")


from langgraph_workflow.build_workflow import build_workflow

def load_reviews_from_file(file_path: Path) -> List[str]:
    """텍스트 파일에서 리뷰 목록을 읽어옵니다."""
    if not file_path.is_file():
        print(f"!!! 경고: 리뷰 파일을 찾을 수 없습니다: {file_path}")
        return ["리뷰 데이터가 없습니다."]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f if line.strip()]
        return reviews
    except Exception as e:
        print(f"!!! 에러: 리뷰 파일 읽기 중 오류 발생: {e}")
        return ["리뷰 데이터를 읽는 데 실패했습니다."]

def run_analysis_pipeline(payload: dict):
    """
    백엔드로부터 받은 데이터(payload)를 기반으로 전체 분석 워크로우를 실행합니다.
    """
    print("--- 분석 파이프라인 시작 ---")
    
    app = build_workflow()

    # user_input을 형식에 맞게 조합
    user_input_str = f"{payload['product_name']}, 예측 기간: {payload['prediction_period']}"
    
    # 파일 경로에서 리뷰 로드
    # SRC_ROOT (AI/src 폴더)의 부모 폴더인 AI 폴더를 기준으로 파일 경로를 계산합니다.
    ai_root = SRC_ROOT.parent
    # 프로젝트 루트는 AI 폴더의 부모 폴더입니다.
    project_root = ai_root.parent
    review_file_path = ai_root / payload['review_file_path']
    reviews = load_reviews_from_file(review_file_path)

    initial_state = {
        "user_input": user_input_str,
        "customer_reviews": "\n".join(reviews)
    }

    final_state = app.invoke(initial_state)
    final_report = final_state.get("final_report", "보고서 생성에 실패했습니다.")
    
    print("--- 분석 파이프라인 종료 ---")
    return final_report


def main():
    """
    백엔드 API 서버를 시뮬레이션하는 메인 실행 함수입니다.
    """
    mock_api_request_payload = {
        "product_name": "Bretford CR4500 Series Slim Rectangular Table", 
        "prediction_period": "다음 1개월",
        "review_file_path": "data/table_reviews.txt"
    }

    final_report = run_analysis_pipeline(mock_api_request_payload)

    print("\n\n\n--- 최종 생성된 보고서 ---")
    print(final_report)


if __name__ == "__main__":
    # pandas에서 날짜 파싱 관련 경고가 나올 수 있으나, 실행에 문제는 없습니다.
    main() 