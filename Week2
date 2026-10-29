"""
AWS ML - Week 02: Data Ingestion, APIs, Vector Embeddings & Semantic Search
=============================================================================
This script demonstrates an end-to-end semantic search pipeline:
1. Data Ingestion & Data Quality:
   - Detecting and handling missing values, duplicate rows, whitespace trimming,
     and category normalization.
2. Web API Integration & JSON Handling:
   - Resilient HTTP requests, JSON parsing, error handling.
3. Text Tokenization & Document Chunking:
   - Character/word chunking with sliding overlap.
4. Dense Vector Embeddings:
   - Generating sentence embeddings with all-MiniLM-L6-v2.
5. NumPy-based Semantic Retrieval:
   - Vectorized cosine similarity computation and top-k ranking.
6. Vector Database (ChromaDB):
   - Persistent storage, document indexing, metadata attachment,
     and structured category filtering.
7. Evaluation & Comparison:
   - Multi-query performance comparison between NumPy cosine similarity and ChromaDB.
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MESSY_DATA_FILE = os.path.join(DATA_DIR, "raw_messy_faqs.csv")
CLEAN_DATA_FILE = os.path.join(DATA_DIR, "faqs.csv")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


# ==============================================================================
# 1. Data Ingestion & Quality Cleaning Pipeline
# ==============================================================================
def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """Loads a CSV dataset and displays initial shape and info."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset from: '{os.path.basename(filepath)}'")
    print(f"       Initial Shape: {df.shape}")
    return df


def clean_faq_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw FAQ data:
    1. Detects and reports null values; removes invalid rows.
    2. Detects and drops duplicate entries.
    3. Trims whitespace and standardizes text casing for category and text fields.
    4. Calculates text length and computes min-max normalized length score.
    """
    print("\n--- [STEP 1: Data Inspection & Cleaning] ---")
    df_clean = df.copy()

    # 1. Missing Values
    null_counts = df_clean.isna().sum()
    total_nulls = null_counts.sum()
    print(f"Total missing values detected: {total_nulls}")
    if total_nulls > 0:
        print("Missing values per column:\n", null_counts[null_counts > 0])
        # Drop rows missing crucial question or answer fields
        df_clean = df_clean.dropna(subset=['question', 'answer'])
        print(f"Rows after dropping nulls: {len(df_clean)}")

    # 2. Text Normalization
    for text_col in ['id', 'question', 'answer', 'category']:
        if text_col in df_clean.columns:
            df_clean[text_col] = df_clean[text_col].astype(str).str.strip()

    if 'category' in df_clean.columns:
        df_clean['category'] = df_clean['category'].str.lower()
        print("Normalized 'category' column to lowercase stripped strings.")

    # 3. Duplicate Handling
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['question', 'answer']).reset_index(drop=True)
    duplicates_removed = initial_len - len(df_clean)
    print(f"Duplicate rows removed: {duplicates_removed}")
    print(f"Final clean dataset count: {len(df_clean)}")

    # 4. Feature Engineering: Normalized Length
    question_lengths = df_clean['question'].str.len().to_numpy()
    min_len = question_lengths.min()
    max_len = question_lengths.max()
    if max_len != min_len:
        df_clean['norm_len'] = (question_lengths - min_len) / (max_len - min_len)
    else:
        df_clean['norm_len'] = 1.0
    print(f"Computed normalized question length (Min: {min_len}, Max: {max_len} chars)")

    return df_clean


# ==============================================================================
# 2. API Request Demonstration
# ==============================================================================
def fetch_sample_api_data(endpoint: str = "https://jsonplaceholder.typicode.com/todos/1") -> dict:
    """
    Demonstrates calling a public REST API, validating status codes, and parsing JSON.
    """
    print("\n--- [STEP 2: API & JSON Handling Demonstration] ---")
    print(f"Sending GET request to: {endpoint}")
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"API Response Status: {response.status_code}")
        print("Sample Payload:", json.dumps(data, indent=2))
        return data
    except requests.RequestException as exc:
        print(f"[WARNING] API request failed (offline environment fallback): {exc}")
        mock_payload = {"userId": 1, "id": 1, "title": "Mock offline task", "completed": False}
        print("Using local mock payload:", mock_payload)
        return mock_payload


# ==============================================================================
# 3. Document Chunking
# ==============================================================================
def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """
    Splits text into chunks of specified character size with sliding overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start += (chunk_size - overlap)

    return chunks


# ==============================================================================
# 4. Dense Vector Embeddings
# ==============================================================================
def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Loads SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer
    print(f"\n[INFO] Loading embedding model: '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("[INFO] Model loaded successfully.")
    return model


def generate_faq_embeddings(df: pd.DataFrame, model) -> tuple:
    """
    Creates composite search text ('Question: ... \\nAnswer: ...') and computes
    normalized dense vector embeddings.
    """
    search_texts = [
        f"Question: {row['question']}\nAnswer: {row['answer']}"
        for _, row in df.iterrows()
    ]
    df_with_search = df.copy()
    df_with_search['search_text'] = search_texts

    print(f"[INFO] Generating embeddings for {len(search_texts)} FAQ documents...")
    embeddings = model.encode(search_texts, normalize_embeddings=True, show_progress_bar=False)
    print(f"[INFO] Embeddings generated with shape: {embeddings.shape}")
    return df_with_search, embeddings


# ==============================================================================
# 5. Pure NumPy Vectorized Cosine Similarity Search
# ==============================================================================
def numpy_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity between a 1D query vector and a 2D matrix of vectors.
    Since embeddings are normalized (L2 norm = 1.0), cosine similarity is the dot product.
    """
    # Dot product calculation
    return np.dot(vec_b, vec_a)


def search_numpy(query: str, df: pd.DataFrame, vectors: np.ndarray, model, top_k: int = 3, category: str = None) -> pd.DataFrame:
    """
    Executes semantic search using pure NumPy vector operations.
    """
    # Apply category filter if requested
    if category is not None:
        mask = (df['category'] == category.strip().lower()).to_numpy()
        candidates_df = df[mask].reset_index(drop=True)
        candidate_vectors = vectors[mask]
        if len(candidates_df) == 0:
            return pd.DataFrame(columns=['id', 'question', 'answer', 'category', 'score'])
    else:
        candidates_df = df
        candidate_vectors = vectors

    # Embed query
    query_vector = model.encode([query], normalize_embeddings=True)[0]

    # Compute cosine similarity
    scores = numpy_cosine_similarity(query_vector, candidate_vectors)

    # Top-K ranking
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = candidates_df.iloc[ranked_indices][['id', 'question', 'answer', 'category']].copy()
    results['score'] = scores[ranked_indices]
    return results.reset_index(drop=True)


# ==============================================================================
# 6. ChromaDB Vector Database Storage & Retrieval
# ==============================================================================
def setup_chromadb(df: pd.DataFrame, vectors: np.ndarray, db_path: str = CHROMA_PATH, collection_name: str = "aws_aastu_faqs"):
    """
    Initializes a persistent ChromaDB vector store and indexes documents with metadata.
    """
    import chromadb
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    # Recreate collection cleanly
    existing = [col.name for col in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "AASTU AWS & Machine Learning FAQ Vector Database"}
    )

    ids = [str(val) for val in df['id'].tolist()]
    docs = df['search_text'].tolist()
    metadatas = [
        {"question": row['question'], "answer": row['answer'], "category": row['category']}
        for _, row in df.iterrows()
    ]
    embeddings_list = vectors.tolist()

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings_list
    )
    print(f"[INFO] ChromaDB collection '{collection_name}' initialized with {collection.count()} items.")
    return collection


def search_chromadb(query: str, collection, model, top_k: int = 3, category: str = None) -> pd.DataFrame:
    """
    Retrieves nearest documents from ChromaDB with optional metadata filtering.
    """
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    where_filter = {"category": category.strip().lower()} if category else None

    query_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    records = []
    if query_results["ids"] and len(query_results["ids"][0]) > 0:
        for i, doc_id in enumerate(query_results["ids"][0]):
            meta = query_results["metadatas"][0][i]
            dist = query_results["distances"][0][i]
            # Convert cosine distance to cosine similarity: similarity = 1 - distance
            similarity = 1.0 - (dist / 2.0) if dist <= 2.0 else 0.0
            records.append({
                "id": doc_id,
                "question": meta["question"],
                "answer": meta["answer"],
                "category": meta["category"],
                "similarity": similarity,
                "distance": dist
            })

    return pd.DataFrame(records)


# ==============================================================================
# 7. Comprehensive Evaluation Pipeline
# ==============================================================================
def run_evaluation(df: pd.DataFrame, vectors: np.ndarray, collection, model):
    """
    Runs multi-query evaluation covering diverse user phrasings, category filters,
    and analyzes an edge-case / poor query.
    """
    print("\n" + "=" * 70)
    print("                     EVALUATION & BENCHMARKS")
    print("=" * 70)

    test_queries = [
        ("How do I store datasets in the cloud?", None),
        ("Where can I train my neural network models?", "aws_cloud"),
        ("Why compare vector angles instead of euclidean distance?", "machine_learning"),
        ("What time do I have to submit my weekly lab assignments?", "academics"),
        ("I forgot my password to login", "support"),
    ]

    print("\n--- Test Set: Multi-Query Search Performance ---")
    for q_text, cat in test_queries:
        filter_str = f" [Category Filter: {cat}]" if cat else ""
        print(f"\n>> QUERY: '{q_text}'{filter_str}")
        res = search_numpy(q_text, df, vectors, model, top_k=2, category=cat)
        for idx, row in res.iterrows():
            print(f"   [{idx + 1}] (Score: {row['score']:.4f}) [{row['category'].upper()}] {row['question']}")

    # Edge Case / Poor Performance Query Analysis
    ambiguous_query = "How do I fix errors?"
    print(f"\n--- Edge Case Analysis: Ambiguous Query ---")
    print(f">> QUERY: '{ambiguous_query}'")
    poor_res = search_numpy(ambiguous_query, df, vectors, model, top_k=3)
    for idx, row in poor_res.iterrows():
        print(f"   [{idx + 1}] (Score: {row['score']:.4f}) [{row['category']}] {row['question']}")
    print("\n[DIAGNOSTIC ANALYSIS]")
    print("• Issue: Query is overly generic without domain context (syntax error? git error? portal login error?).")
    print("• Solution: Implement query disambiguation, context expansion, or enforce category filtering.")


# ==============================================================================
# 8. Main Entrypoint
# ==============================================================================
def main():
    print("=" * 70)
    print("  AWS ML - Week 02: Semantic Search & Vector DB Pipeline")
    print("=" * 70)

    # 1. Load data
    target_data_file = MESSY_DATA_FILE if os.path.exists(MESSY_DATA_FILE) else CLEAN_DATA_FILE
    df_raw = load_and_inspect_data(target_data_file)

    # 2. Clean data
    df_clean = clean_faq_dataset(df_raw)

    # 3. API demo
    fetch_sample_api_data()

    # 4. Chunking demo
    sample_text = (
        "Amazon SageMaker is a fully managed machine learning service. "
        "With SageMaker, data scientists and developers can quickly build, train, "
        "and deploy models into a production-ready hosted environment. "
        "It provides managed instances running custom or pre-built ML containers."
    )
    chunks = chunk_text(sample_text, chunk_size=90, overlap=20)
    print(f"\n--- [STEP 3: Chunking Demonstration] ---\nOriginal length: {len(sample_text)} characters")
    for i, chk in enumerate(chunks):
        print(f"Chunk {i + 1} ({len(chk)} chars): '{chk}'")

    # 5. Embeddings
    model = get_embedding_model()
    df_embedded, vectors = generate_faq_embeddings(df_clean, model)

    # 6. ChromaDB Setup
    collection = setup_chromadb(df_embedded, vectors)

    # 7. Evaluation & Comparison
    run_evaluation(df_embedded, vectors, collection, model)

    print("\n[SUCCESS] Week 02 Semantic Search Pipeline executed flawlessly!")


if __name__ == "__main__":
    main()
