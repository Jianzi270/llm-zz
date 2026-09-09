"""层次化知识库检索接口（模块 B5/C2）：按"类别 → 文档 → 文本块"三级检索（DC-RAG）。

用法：
  python -m src.retrieval.kb "深圳2026年经济社会发展目标" [--top-c 2] [--top-d 3] [--top-k 3]
  from src.retrieval.kb import load_kb, retrieve

三级检索流程：
  1. 类别级：查询向量与类别代表向量（category_vectors）比对，选 top C 类别
  2. 文档级：在选中类别内与文档代表向量（doc_vectors）比对，选 top D 文档
  3. 文本块级：在选中文档内与文本块向量（chunk_vectors）比对，选 top K 块
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"

_kb = None


def load_kb() -> dict:
    """加载知识库（向量 + 三级索引 + 嵌入模型），懒加载。"""
    global _kb
    if _kb is None:
        loaded = {
            "chunk_vectors": np.load(KB_DIR / "chunk_vectors.npy"),
            "doc_vectors": np.load(KB_DIR / "doc_vectors.npy"),
            "category_vectors": np.load(KB_DIR / "category_vectors.npy"),
            "index": json.loads((KB_DIR / "index.json").read_text(encoding="utf-8")),
        }
        idx = loaded["index"]
        expected = (len(idx.get("chunks", [])), len(idx.get("doc_ids", [])))
        actual = (loaded["chunk_vectors"].shape[0], loaded["doc_vectors"].shape[0])
        if actual != expected:
            raise RuntimeError(f"知识库索引与向量数量不一致: index={expected}, vectors={actual}")
        _kb = loaded
    return _kb


def _topk(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


def _select_diverse_chunks(chunk_scores: list[tuple[int, float]], chunks: list[dict],
                           top_k: int, max_per_doc: int = 2) -> list[tuple[int, float]]:
    """按分数选择文本块，并限制单个文档垄断全部上下文。"""
    selected = []
    doc_counts: dict[str, int] = {}
    for item in sorted(chunk_scores, key=lambda value: value[1], reverse=True):
        doc_id = chunks[item[0]]["doc_id"]
        if doc_counts.get(doc_id, 0) >= max_per_doc:
            continue
        selected.append(item)
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


REGION_ALIASES = {
    "国务院": "全国", "全国": "全国", "广东": "广东省", "深圳": "深圳市",
    "南山": "南山区", "福田": "福田区", "罗湖": "罗湖区", "龙岗": "龙岗区",
    "宝安": "宝安区", "龙华": "龙华区", "光明": "光明区", "坪山": "坪山区",
    "盐田": "盐田区", "大鹏": "大鹏新区", "深汕": "深汕特别合作区",
}


def extract_metadata_filters(query: str) -> dict[str, str]:
    """从查询中提取确定性较高的年份和地区约束。"""
    year = re.search(r"(?:19|20)\d{2}", query)
    region = ""
    # 先匹配区级，避免“深圳市南山区”被提前识别为市级。
    for alias in sorted(REGION_ALIASES, key=len, reverse=True):
        if alias in query:
            region = REGION_ALIASES[alias]
            if region not in {"深圳市", "广东省", "全国"}:
                break
    return {"year": year.group(0) if year else "", "region": region}


def _filter_doc_ids(doc_ids: set[str], index: dict, filters: dict[str, str]) -> set[str]:
    metadata = index.get("doc_metadata", {})
    if not metadata or not any(filters.values()):
        return doc_ids
    matched = {
        doc for doc in doc_ids
        if (not filters["year"] or str(metadata.get(doc, {}).get("year", "")) == filters["year"])
        and (not filters["region"] or metadata.get(doc, {}).get("region", "") == filters["region"])
    }
    return matched


def retrieve(query: str, top_c: int = 2, top_d: int = 3, top_k: int = 3) -> list[dict]:
    """三级检索：类别 → 文档 → 文本块，返回命中的文本块列表（含元数据与相似度）。"""
    from src.embed.embed import embed_queries
    kb = load_kb()
    q = embed_queries([query])[0]

    idx = kb["index"]
    cat_vecs, doc_vecs, chunk_vecs = kb["category_vectors"], kb["doc_vectors"], kb["chunk_vectors"]

    # 1. 类别级检索
    cat_scores = cat_vecs @ q
    cat_hits = _topk(cat_scores, top_c)

    # 2. 文档级检索：在选中类别内
    cand_docs = set()
    for cid, _ in cat_hits:
        cand_docs.update(idx["cluster_docs"][str(cid)])
    filters = extract_metadata_filters(query)
    filtered = _filter_doc_ids(cand_docs, idx, filters)
    # 若类别路由漏掉了精确年份/地区文档，则扩大到全库的元数据匹配结果。
    if any(filters.values()) and not filtered:
        filtered = _filter_doc_ids(set(idx["doc_ids"]), idx, filters)
    cand_docs = filtered or cand_docs
    doc_positions = {doc: i for i, doc in enumerate(idx["doc_ids"])}
    doc_scores = {d: float(doc_vecs[doc_positions[d]] @ q) for d in cand_docs}
    doc_hits = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_d]

    # 3. 文本块级检索：在选中文档内
    cand_chunks = [(i, c) for i, c in enumerate(idx["chunks"]) if c["doc_id"] in dict(doc_hits)]
    if not cand_chunks:
        return []
    chunk_scores = [(i, float(chunk_vecs[i] @ q)) for i, _ in cand_chunks]
    chunk_hits = _select_diverse_chunks(chunk_scores, idx["chunks"], top_k)

    results = []
    doc_cluster = idx["doc_cluster"]
    for i, score in chunk_hits:
        c = idx["chunks"][i]
        results.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "cluster": doc_cluster.get(c["doc_id"]),
            "score": round(score, 4),
            "text": c["text"],
            "doc_title": c.get("doc_title", c["doc_id"]),
            "source_level": c.get("source_level", ""),
            "region": c.get("region", ""),
            "year": c.get("year", ""),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="层次化知识库检索（DC-RAG 三级）")
    parser.add_argument("query", help="查询问题")
    parser.add_argument("--top-c", type=int, default=2, help="候选类别数")
    parser.add_argument("--top-d", type=int, default=3, help="候选文档数")
    parser.add_argument("--top-k", type=int, default=3, help="返回文本块数")
    args = parser.parse_args()

    load_kb()
    print(f"查询: {args.query}")
    print(f"知识库: 类别={len(load_kb()['category_vectors'])}, "
          f"文档={len(load_kb()['doc_vectors'])}, 文本块={len(load_kb()['chunk_vectors'])}")
    print("=" * 60)
    for r in retrieve(args.query, args.top_c, args.top_d, args.top_k):
        print(f"[score={r['score']:.3f} | 类{r['cluster']} | {r['doc_id']}]")
        print(f"  {r['text'][:100]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
