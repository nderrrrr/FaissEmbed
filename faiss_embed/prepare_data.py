import os
import json
import argparse
import numpy as np
from typing import List, Tuple

from faiss_embed.embedding_model import EmbeddingModel

def load_dialogue_json(file_path: str) -> List[List[List[str]]]:
    """
    讀取 JSON 檔並回傳對話資料
    JSON 結構假設為 List of dialogues,
    每個 dialogue 為 List of utterances,
    每個 utterance 為 [原住民語, 中文]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tribe_code", required=True, type=str, help="例如 CAA, 代表海岸阿美語")
    parser.add_argument("--model_name", default="BAAI/bge-m3", type=str, help="可選擇其他模型")
    args = parser.parse_args()

    data_dir = os.path.join("data", args.tribe_code)
    dialogue_file = os.path.join(data_dir, f"{args.tribe_code}_dialogue.json")
    
    # 讀取對話資料
    dialogues = load_dialogue_json(dialogue_file)

    # 產生句子清單 & 索引
    sentence_list = []
    sentence_index_map = []
    for d_id, dialogue in enumerate(dialogues):
        for u_id, utterance in enumerate(dialogue):
            # utterance = ["原住民語", "中文"]
            zh_sentence = utterance[1]
            sentence_list.append(zh_sentence)
            sentence_index_map.append((d_id, u_id))

    # 初始化模型
    model = EmbeddingModel(model_name=args.model_name)
    # 產生 embedding
    embedding_dict = model.get_multiple_embeddings(sentence_list)
    # 順序對齊
    embedding_list = [embedding_dict[sent] for sent in sentence_list]

    # 輸出 .npy 檔
    emb_output_path = os.path.join(data_dir, f"{args.tribe_code}_embedding_list.npy")
    idx_output_path = os.path.join(data_dir, f"{args.tribe_code}_sentence_index_map.npy")

    np.save(emb_output_path, np.array(embedding_list, dtype=object))
    np.save(idx_output_path, np.array(sentence_index_map, dtype=object))

    print(f"[INFO] Embeddings saved to: {emb_output_path}")
    print(f"[INFO] Index mapping saved to: {idx_output_path}")

if __name__ == "__main__":
    main()
