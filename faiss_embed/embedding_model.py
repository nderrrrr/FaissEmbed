import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

class EmbeddingModel:
    """
    EmbeddingModel 使用指定的 Transformer (如 BERT) 將句子向量化。
    """
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)

    def get_single_embedding(self, sentence: str) -> np.ndarray:
        """
        使用 BGEM3FlagModel 的 encode 方法取得向量 (dense_vecs)。
        回傳 shape 為 (dim,) 的 numpy array。
        """
        result = self.model.encode([sentence])  # encode 一個 list
        
        dense_vecs = result['dense_vecs']       # 取 'dense_vecs'
        return dense_vecs[0]  

    def get_multiple_embeddings(self, sentences: list) -> dict:
        """
        將多句子轉成 dict: {原句: embedding向量}
        """
        embeddings = {}
        total = len(sentences)
        progress_bar = tqdm(total=total, desc="Loading sentences embeddings", dynamic_ncols=False)

        # 這裡範例直接單句 encode，可自行優化成多句一起 encode 以提高效率
        update_interval = max(1, total // 10) if total > 0 else 1

        for i, sentence in enumerate(sentences):
            result = self.get_single_embedding(sentence) 
            embeddings[sentence] = result

            if (i + 1) % update_interval == 0 or i == total - 1:
                progress_bar.update(update_interval)

        progress_bar.close()
        return embeddings
