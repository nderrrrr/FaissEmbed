# Faiss Embed

本專案展示如何結合 [**BGE-M3**](https://huggingface.co/BAAI/bge-m3) 向量模型與 [**Faiss**](https://github.com/facebookresearch/faiss) 向量檢索引擎，實作對**原住民族語中文對話語料**的 dense retrieval 功能。

---

## 🔍 專案特色

- 使用 **BAAI/bge-m3** 模型產生 dense embedding，可支援多語、多層次語意理解。
- 搭配 **Faiss IndexFlatL2** 做最近鄰搜尋（向量相似度）。
- 支援多族語對話語料，自動處理向量化與檢索索引建構。
- 向量與對應對話位置會預先儲存為 `.npy`，大幅提升查詢效率。

---

## 🧠 使用模型

本專案採用 [**BGE-M3** 模型](https://huggingface.co/BAAI/bge-m3)，特點包括：

- 輸出 dense vectors，適合語意檢索與語句相似度比較。
- 支援長文本（最多 8192 tokens）。
- 多語言訓練，適合應用於中文語料。
- 使用 [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) 套件封裝，簡單高效。

---

## 📁 資料簡介

我們處理了以下五種原住民族語對話語料（族語與中文平行句），資料皆位於 `./data/` 目錄下：

| Tribe Code | 中文名稱     | 原住民族語名稱      |
|------------|--------------|---------------------|
| NSA        | 南勢阿美     | Southern Amis       |
| CAA        | 海岸阿美     | Coastal Amis        |
| WDT        | 萬大泰雅     | Wanda Tayal         |
| SJT        | 四季泰雅     | Siji Tayal          |
| DDS        | 都達賽德克   | Duda Seediq         |

> 📂 `data/` 資料夾已加入 `.gitignore`，請手動放置你擁有的對話語料。

---

## 📦 安裝方式

建議在虛擬環境中安裝：

```bash
pip install -r requirements.txt
```

## 🏗️ 資料預處理流程
請將每一族語的對話資料（JSON 格式）放置於：
```bash
./data/{tribe_code}/{tribe_code}_dialogue.json
```
每段對話為一組句子對應結構：
```json
[
    [
        ["族語1", "你好"],
        ["族語2", "你今天好嗎？"]
    ],
    [
        ["族語3", "吃飯了嗎？"],
        ["族語4", "還沒，你呢？"]
    ]
]
```

執行以下指令產生該族語的 embedding 與對應索引檔案：
```bash
python faiss_embed/prepare_data.py --tribe_code CAA
```
完成後會在 `./data/CAA/` 中產生：
- `CAA_embedding_list.npy`：所有中文句子的向量表示
- `CAA_sentence_index_map.npy`：每個句子對應在原始對話中的位置 (dialogue_id, utterance_id)

## 🔍 查詢與檢索範例
`usage_example.ipynb` 提供一個範例查詢流程：
1. 載入 `.npy` 向量與位置索引
2. 將查詢句子轉為向量
3. 在 Faiss 索引中找最相似句子
4. 回傳其在原始對話中的「下一句」作為回應建議

輸出範例：
```
查詢句子: 今天天氣如何？
回傳下一句: 太陽出來了，天氣很熱。
```

## 🧠 技術細節
使用 BGE-M3 模型產生句子向量：
```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3")
vecs = model.encode(["我今天很好"])["dense_vecs"]
```

使用 Faiss 建立與查詢向量索引：
```python
import faiss
index = faiss.IndexFlatL2(dim)
index.add(embedding_list)
D, I = index.search(query_vec, k=1)
```

## 📂 專案結構
```bash
FaissEmbed/
├── faiss_embed/               # 核心模組
│   ├── embedding_model.py     # BGE-M3 向量模型封裝
│   ├── faiss_indexer.py       # 建立與查詢 Faiss 索引
│   ├── prepare_data.py        # 轉換資料並儲存 .npy 向量與索引
├── data/                      # 語料資料（需手動放入，已忽略 Git）
│   └── CAA/...
├── usage_example.py           # 查詢範例：輸入句子 → 回傳下一句
├── requirements.txt
└── README.md
```