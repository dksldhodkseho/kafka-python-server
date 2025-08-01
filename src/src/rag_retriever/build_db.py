import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import gc
import torch
import os
from pathlib import Path

# --- 설정 ---
# 프로젝트 루트 경로를 시스템에 맞게 명시적으로 지정합니다.
# 사용자님의 환경에 맞게 경로 구분자를 사용합니다.
PROJECT_ROOT = Path("C:/Users/User/Desktop/KT_AIVLE_BigProject")

# CSV 파일 경로
CSV_FILE_PATH = PROJECT_ROOT / "AI" / "data" / "news_with_keywords_global.csv"
# ChromaDB 저장 경로
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"
# 사용할 임베딩 모델
EMBEDDING_MODEL = 'upskyy/bge-m3-korean'
# ChromaDB 컬렉션 이름
COLLECTION_NAME = "news_trend"

def build_database():
    """
    CSV 파일에서 데이터를 읽어 ChromaDB 벡터 데이터베이스를 구축합니다.
    """
    print("--- ChromaDB 데이터베이스 구축 시작 ---")
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"ChromaDB 저장 위치: {PERSIST_DIRECTORY}")

    # 1. 저장 디렉토리 명시적 생성
    try:
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        print(f"성공: 디렉토리 생성 완료 또는 이미 존재함 - {PERSIST_DIRECTORY}")
    except Exception as e:
        print(f"치명적 오류: 디렉토리 생성 실패 - {e}")
        return

    # 2. 임베딩 모델 준비
    print(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # 3. CSV 데이터 불러오기
    print(f"CSV 파일 로딩: {CSV_FILE_PATH}")
    if not os.path.exists(CSV_FILE_PATH):
        print(f"오류: {CSV_FILE_PATH}를 찾을 수 없습니다.")
        return

    df = pd.read_csv(CSV_FILE_PATH)
    df.drop_duplicates(subset=["desc", "content", "keyword", "global_keywords"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"데이터 로딩 및 정제 완료. {len(df)}개 행")

    # 4. 크로마DB 클라이언트 및 컬렉션 준비
    print(f"ChromaDB 클라이언트 초기화 시작...")
    # 데이터를 디스크에 안정적으로 저장하기 위해 PersistentClient를 명시적으로 사용합니다.
    client = chromadb.PersistentClient(path=str(PERSIST_DIRECTORY))
    
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"기존 컬렉션 '{COLLECTION_NAME}' 삭제 중...")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"'{COLLECTION_NAME}' 컬렉션 생성 완료.")

    # 5. 데이터 준비
    texts = []
    metadatas = []
    ids = []
    for idx, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        media = str(row.get("media", "")).strip()
        date = str(row.get("date", "")).strip()
        for field, doc_type in [("desc", "desc"), ("content", "content"), ("keyword", "keyword"), ("global_keywords", "global_kw")]:
            text = str(row.get(field, "")).strip()
            if text and text.lower() != "nan":
                texts.append(text)
                metadatas.append({"type": doc_type, "title": title, "media": media, "date": date})
                ids.append(f"{doc_type}_{idx}")
    
    if not texts:
        print("오류: DB에 추가할 텍스트 데이터가 없습니다.")
        return

    # 6. 임베딩 및 적재
    print(f"{len(texts)}개 텍스트 임베딩 및 적재 시작...")
    try:
        vectors = embedder.encode(texts, batch_size=32, show_progress_bar=True)
        collection.add(documents=texts, embeddings=[vec.tolist() for vec in vectors], metadatas=metadatas, ids=ids)
        print("✅ ChromaDB 벡터DB 적재 완료!")
    except Exception as e:
        print(f"DB 적재 중 오류 발생: {e}")
    finally:
        # 7. 메모리 정리
        del embedder, vectors, texts, metadatas, ids, df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("메모리 정리 완료.")

if __name__ == "__main__":
    build_database()
