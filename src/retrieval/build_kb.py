"""层次化知识库构建（模块 B5）：构建"类别 → 文档 → 文本块"三级树形知识库。

流程：
  1. 读取文本块（chunks.jsonl）与聚类结果（clusters.jsonl，软概率）
  2. 用 BGE 嵌入模型向量化全部文本块
  3. 文档代表向量 = 文档内文本块向量均值
  4. 类别代表向量 = 类内文档向量按软概率加权均值
  5. 保存三级索引与向量到 data/knowledge_base/（不入库，可重建）

用法：
  python -m src.retrieval.build_kb

输出（data/knowledge_base/）：
  chunk_vectors.npy       # (15509, 512) 文本块向量
  doc_vectors.npy         # (160, 512) 文档代表向量
  category_vectors.npy    # (K, 512) 类别代表向量
  index.json              # 三级索引（类别->文档->块 映射与文本）
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
CLUSTERS_FILE = PROJECT_ROOT / "data" / "processed" / "clusters.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def main():
    # 1. 加载文本块
    chunks = [json.loads(l) for l in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"文本块: {len(chunks)}")

    # 2. 加载聚类（doc_id -> {cluster, prob})
    clusters = {}
    n_clusters = 0
    for line in CLUSTERS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        clusters[d["doc_id"]] = d
        n_clusters = max(n_clusters, d["cluster"] + 1)
    print(f"文档聚类: {len(clusters)} 篇, {n_clusters} 类")

    # 3. 向量化文本块（若缓存块数与当前一致则复用，避免重复耗时；不一致说明有增量数据，需重算）
    from src.embed.embed import embed_texts
    chunk_vec_file = OUT_DIR / "chunk_vectors.npy"
    if chunk_vec_file.exists():
        cached = np.load(chunk_vec_file)
        if cached.shape[0] == len(chunks):
            chunk_vecs = cached
            print(f"加载缓存文本块向量: {chunk_vecs.shape}")
        else:
            print(f"缓存块数不匹配（缓存 {cached.shape[0]} vs 当前 {len(chunks)}），重新向量化")
            chunk_vecs = embed_texts([c["text"] for c in chunks])
            print(f"文本块向量: {chunk_vecs.shape}")
    else:
        chunk_vecs = embed_texts([c["text"] for c in chunks])
        print(f"文本块向量: {chunk_vecs.shape}")

    # 4. 文档代表向量：使用 LLM 摘要向量（比块均值更能代表文档主题）
    doc_chunk_ids = defaultdict(list)
    for i, c in enumerate(chunks):
        doc_chunk_ids[c["doc_id"]].append(i)
    doc_ids = sorted(doc_chunk_ids)
    summaries = {}
    for line in (PROJECT_ROOT / "data" / "processed" / "summaries.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            summaries[d["doc_id"]] = d["summary"]
    doc_vecs = embed_texts([summaries.get(doc, "") for doc in doc_ids])
    print(f"文档向量（摘要）: {doc_vecs.shape}")

    # 5. 类别代表向量（类内文档按软概率加权均值）
    prob = np.zeros((len(doc_ids), n_clusters))
    for j, doc in enumerate(doc_ids):
        cl = clusters.get(doc, {})
        for cid, p in cl.get("top_probs", {}).items():
            prob[j][int(cid)] = p
        if cl and "top_probs" not in cl:
            prob[j][cl.get("cluster", 0)] = 1.0
    cat_vecs = np.zeros((n_clusters, doc_vecs.shape[1]))
    for c in range(n_clusters):
        w = prob[:, c]
        if w.sum() > 0:
            cat_vecs[c] = (doc_vecs * w[:, None]).sum(axis=0) / w.sum()
    print(f"类别向量: {cat_vecs.shape}")

    # 6. 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "chunk_vectors.npy", chunk_vecs)
    np.save(OUT_DIR / "doc_vectors.npy", doc_vecs)
    np.save(OUT_DIR / "category_vectors.npy", cat_vecs)
    index = {
        "doc_ids": doc_ids,
        "doc_cluster": {d: clusters.get(d, {}).get("cluster", 0) for d in doc_ids},
        "cluster_docs": {str(c): [d for d in doc_ids if clusters.get(d, {}).get("cluster", 0) == c]
                         for c in range(n_clusters)},
        "chunks": [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "text": c["text"]} for c in chunks],
    }
    with (OUT_DIR / "index.json").open("w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False)

    sizes = {str(c): len(index["cluster_docs"][str(c)]) for c in range(n_clusters)}
    print(f"类别规模: {sizes}")
    print(f"知识库构建完成，输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
