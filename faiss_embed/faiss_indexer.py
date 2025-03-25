import faiss
import numpy as np

def embeddings_to_faiss(embeddings_list: list) -> faiss.Index:
    """
    接收 embedding 的 list (皆為 np.array)，建構 FAISS 索引 (IndexFlatL2)
    """
    np_embeddings = np.vstack(embeddings_list)
    embedding_dim = np_embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np_embeddings)
    return index

def search_similar_embeddings(
    query_embedding: np.ndarray,
    faiss_index: faiss.Index,
    top_k: int = 1
):
    """
    在已建立的 Faiss 索引中搜尋最相似的 embedding
    回傳 (distances, indices)
    """
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    D, I = faiss_index.search(query_embedding, top_k)
    return D, I
