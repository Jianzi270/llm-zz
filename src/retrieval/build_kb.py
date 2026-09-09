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
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
CLUSTERS_FILE = PROJECT_ROOT / "data" / "processed" / "clusters.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def _chunk_cache_fingerprint(chunks: list[dict]) -> str:
    """计算会影响文本块向量的稳定指纹。"""
    embed_cfg = (PROJECT_ROOT / "src" / "embed" / "config.json").read_bytes()
    digest = hashlib.sha256(embed_cfg)
    for chunk in chunks:
        digest.update(chunk["chunk_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk["text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _embed_config_fingerprint() -> str:
    return hashlib.sha256((PROJECT_ROOT / "src" / "embed" / "config.json").read_bytes()).hexdigest()


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

    # 3. 向量化文本块：全量指纹一致则直接复用；否则按 chunk_id+文本复用未变化向量。
    from src.embed.embed import embed_texts
    chunk_vec_file = OUT_DIR / "chunk_vectors.npy"
    manifest_file = OUT_DIR / "manifest.json"
    fingerprint = _chunk_cache_fingerprint(chunks)
    embed_config_fingerprint = _embed_config_fingerprint()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    embed_dim = int(json.loads(
        (PROJECT_ROOT / "src" / "embed" / "config.json").read_text(encoding="utf-8"))["dim"])
    if chunk_vec_file.exists() and manifest.get("chunk_fingerprint") == fingerprint:
        cached = np.load(chunk_vec_file)
        if cached.shape == (len(chunks), embed_dim):
            chunk_vecs = cached
            print(f"加载缓存文本块向量: {chunk_vecs.shape}")
        else:
            print(f"缓存形状不匹配（缓存 {cached.shape}），重新向量化")
            chunk_vecs = embed_texts([c["text"] for c in chunks])
            print(f"文本块向量: {chunk_vecs.shape}")
    else:
        old_index_file = OUT_DIR / "index.json"
        can_reuse = (chunk_vec_file.exists() and old_index_file.exists()
                     and manifest.get("embed_config_fingerprint") in (None, embed_config_fingerprint))
        cached = np.load(chunk_vec_file) if can_reuse else None
        old_chunks = json.loads(old_index_file.read_text(encoding="utf-8")).get("chunks", []) if can_reuse else []
        if cached is not None and cached.shape == (len(old_chunks), embed_dim):
            old_positions = {(c["chunk_id"], c["text"]): i for i, c in enumerate(old_chunks)}
            chunk_vecs = np.empty((len(chunks), embed_dim), dtype=np.float32)
            missing_positions = []
            for i, chunk in enumerate(chunks):
                old_pos = old_positions.get((chunk["chunk_id"], chunk["text"]))
                if old_pos is None:
                    missing_positions.append(i)
                else:
                    chunk_vecs[i] = cached[old_pos]
            if missing_positions:
                fresh = embed_texts([chunks[i]["text"] for i in missing_positions])
                chunk_vecs[missing_positions] = fresh
            print(f"增量向量化: 复用 {len(chunks) - len(missing_positions)} 块，新算 {len(missing_positions)} 块")
        else:
            if chunk_vec_file.exists():
                print("嵌入配置变化或旧索引不兼容，重新向量化全部文本块")
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
    missing_summaries = [doc for doc in doc_ids if not summaries.get(doc, "").strip()]
    if missing_summaries:
        raise RuntimeError(f"以下入库文档缺少摘要: {missing_summaries[:5]}")
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
            norm = np.linalg.norm(cat_vecs[c])
            if norm:
                cat_vecs[c] /= norm
    print(f"类别向量: {cat_vecs.shape}")

    # 6. 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "chunk_vectors.npy", chunk_vecs)
    np.save(OUT_DIR / "doc_vectors.npy", doc_vecs)
    np.save(OUT_DIR / "category_vectors.npy", cat_vecs)
    # 软聚类倒排：一个文档可以进入多个类别；低概率噪声不纳入候选。
    cluster_docs = {}
    cluster_doc_weights = {}
    for c in range(n_clusters):
        weighted = [(doc, float(prob[j, c])) for j, doc in enumerate(doc_ids) if prob[j, c] >= 0.05]
        weighted.sort(key=lambda item: item[1], reverse=True)
        cluster_docs[str(c)] = [doc for doc, _ in weighted]
        cluster_doc_weights[str(c)] = {doc: round(weight, 6) for doc, weight in weighted}

    doc_metadata = {}
    for chunk in chunks:
        doc_metadata.setdefault(chunk["doc_id"], {
            "doc_title": chunk.get("doc_title", ""),
            "source_level": chunk.get("source_level", ""),
            "region": chunk.get("region", ""),
            "year": str(chunk.get("year", "")),
        })

    index = {
        "doc_ids": doc_ids,
        "doc_metadata": doc_metadata,
        "doc_cluster": {d: clusters.get(d, {}).get("cluster", 0) for d in doc_ids},
        "cluster_docs": cluster_docs,
        "cluster_doc_weights": cluster_doc_weights,
        "chunks": [{
            "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "text": c["text"],
            "doc_title": c.get("doc_title", ""), "source_level": c.get("source_level", ""),
            "region": c.get("region", ""), "year": str(c.get("year", "")),
        } for c in chunks],
    }
    with (OUT_DIR / "index.json").open("w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False)
    manifest_file.write_text(json.dumps({
        "chunk_fingerprint": fingerprint,
        "embed_config_fingerprint": embed_config_fingerprint,
        "chunk_count": len(chunks),
        "doc_count": len(doc_ids),
        "category_count": n_clusters,
        "vector_dim": int(chunk_vecs.shape[1]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    sizes = {str(c): len(index["cluster_docs"][str(c)]) for c in range(n_clusters)}
    print(f"类别规模: {sizes}")
    print(f"知识库构建完成，输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
