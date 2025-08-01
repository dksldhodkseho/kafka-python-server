import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import torch
import gc
import logging
from dotenv import load_dotenv

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# --- 상수 정의 ---
EMBEDDING_MODEL = 'upskyy/bge-m3-korean'
COLLECTION_NAME = "news_trend"

class RagRetriever:
    """
    ChromaDB에 저장된 벡터를 기반으로 시장 및 트렌드 정보를 검색하고 요약합니다.
    """
    def __init__(self, project_root: str):
        logger.info("RagRetriever 초기화 시작...")
        
        # project_root는 'KT_AIVLE_BigProject'의 전체 경로입니다.
        # build_db.py가 프로젝트 루트에 'chroma_db'를 생성하므로, 그 경로를 그대로 사용합니다.
        persist_directory = os.path.join(project_root, "chroma_db")

        if not os.path.exists(persist_directory):
            error_msg = f"ChromaDB 디렉토리를 찾을 수 없습니다: {persist_directory}. 'AI/src/rag_retriever/build_db.py'를 먼저 실행하여 데이터베이스를 구축하세요."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            self.client = chromadb.Client(Settings(persist_directory=persist_directory, anonymized_telemetry=False))
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
            logger.info(f"'{COLLECTION_NAME}' 컬렉션에 성공적으로 연결했습니다.")
            
            logger.info(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
            
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            logger.info("RagRetriever 초기화 완료.")
        except Exception as e:
            logger.error(f"RagRetriever 초기화 중 오류 발생: {e}", exc_info=True)
            raise

    def retrieve_and_summarize(self, query: str) -> str:
        """
        사용자 쿼리를 기반으로 관련 문서를 검색하고, LLM을 통해 요약된 답변을 생성합니다.
        """
        logger.info(f"RAG 검색 시작: '{query}'")

        try:
            # 1. 사용자 쿼리 임베딩
            q_vec = self.embedder.encode([query])[0].tolist()

            # 2. ChromaDB에서 유사 문서 검색
            results = self.collection.query(query_embeddings=[q_vec], n_results=10)

            if not results or not results.get('documents') or not results['documents'][0]:
                logger.warning("관련 시장 동향 데이터를 찾을 수 없습니다.")
                return "관련 시장 동향 데이터를 찾을 수 없습니다."

            # 3. 결과 후처리 및 LLM 프롬프트 준비
            ref_docs = []
            seen_titles = set()
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                title = meta.get('title', '').strip()
                media = meta.get('media', '').strip()
                
                # 내용이 유사한 다른 기사를 필터링하기 위해 제목+매체 조합으로 중복 체크
                if not title or (title, media) in seen_titles:
                    continue
                
                ref_docs.append({'doc': doc, 'title': title, 'media': media})
                seen_titles.add((title, media))
                
                # LLM 컨텍스트 길이를 고려하여 최대 5개의 고유한 기사만 사용
                if len(ref_docs) >= 5:
                    break
            
            if not ref_docs:
                logger.warning("유효한 근거 문서를 찾지 못했습니다.")
                return "관련 시장 동향 데이터를 찾을 수 없습니다."

            # 4. LLM을 위한 프롬프트 생성
            context_docs = ""
            for i, ref in enumerate(ref_docs, 1):
                context_docs += f"[{i}] {ref['doc']}\n\n"
            
            # 제품명만 추출하여 질문 구성
            product_name = query.split(',')[0].strip()
            final_question = f"다음 제품과 관련된 최신 시장 동향, 소비자 인식, 기술 트렌드를 주어진 문서들을 바탕으로 분석하고 요약해줘: {product_name}"

            system_prompt = (
                "너는 시장 분석 전문가야. 주어진 컨텍스트 문서들을 바탕으로 질문에 대해 종합적으로 분석하고 간결한 보고서 형식으로 요약해야 한다.\n"
                "답변은 반드시 주어진 문서 내용에만 근거해야 하며, 너의 사전 지식을 사용해서는 안 된다.\n"
                "만약 문서에 질문과 관련된 내용이 없다면, '제공된 문서에서 관련 정보를 찾을 수 없습니다.'라고 명확히 답변해야 한다.\n"
                "답변 마지막에는 '[근거 자료]' 섹션을 만들어, 참고한 모든 문서의 '매체명: 기사제목'을 각 줄에 하나씩 나열해야 한다."
            )
            
            user_prompt = f"## 컨텍스트 문서:\n{context_docs}\n## 질문:\n{final_question}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # 5. LLM 호출 및 결과 생성
            logger.info("LLM을 호출하여 트렌드 요약 보고서 생성 중...")
            response = self.llm.invoke(messages)
            
            summary = response.content
            sources = "\n\n[근거 자료]"
            for ref in ref_docs:
                sources += f"\n- {ref['media']}: {ref['title']}"
            
            final_response = summary + sources
            logger.info("트렌드 요약 보고서 생성 완료.")
            return final_response

        except Exception as e:
            logger.error(f"RAG 처리 중 오류 발생: {e}", exc_info=True)
            return f"시장 동향 분석 중 오류가 발생했습니다: {e}"

    def __del__(self):
        """
        객체 소멸 시 메모리 정리
        """
        logger.info("RagRetriever 리소스 정리 중...")
        del self.embedder, self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 