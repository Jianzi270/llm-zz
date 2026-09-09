"""C1 输入增强 + C2 DC-RAG 检索完整链路（本地验证）。

流程：
  C1 输入增强：用 LLM（OpenAI 兼容，复用 .env 配置）将用户问题改写为适合语义检索的
              查询（补全上下文、提取主题/地域/时间），失败时回退原文；随后用 BGE 向量化。
  C2 DC-RAG 检索：按"类别 → 文档 → 文本块"三级检索，返回最相关文本块；
              同时给出扁平检索（全库 top-k）结果作对比，验证 DC-RAG 效果。

用法：
  python -m src.retrieval.query_pipeline "深圳2026年经济社会发展目标" [--top-c 2] [--top-d 3] [--top-k 3]
"""
import argparse
import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.kb import load_kb, _topk, retrieve  # noqa: E402

ENHANCE_PROMPT = (
    "你是政务信息检索助手。用户输入一个查询，请将其改写为更适合语义检索的形式，要求：\n"
    "1. 补全上下文与完整表述（如缩写、年份、简称）；\n"
    "2. 明确检索意图（提取主题、地域、时间、对象等要素）；\n"
    "3. 直接输出改写后的查询，不要任何解释或前缀。\n"
    "原问题：{question}\n"
)


def enhance_query(question: str) -> str:
    """C1：LLM 输入增强，失败时回退原文。"""
    import os
    from src.summarize.summarize import get_llm_settings
    cfg = json.loads((PROJECT_ROOT / "src" / "summarize" / "config.json").read_text(encoding="utf-8"))
    key, base_url, model = get_llm_settings(cfg)
    if not key:
        print("[C1] 未配置 API Key，跳过增强，使用原问题")
        return question
    try:
        resp = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是专业的政务信息检索助手。"},
                    {"role": "user", "content": ENHANCE_PROMPT.format(question=question)},
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=60,
        )
        resp.raise_for_status()
        enhanced = resp.json()["choices"][0]["message"].get("content", "").strip()
        return enhanced or question
    except Exception as e:
        print(f"[C1] 增强失败（{e}），使用原问题")
        return question


def dc_rag_retrieve(question: str, top_c: int = 2, top_d: int = 3, top_k: int = 3) -> list[dict]:
    """C2：DC-RAG 三级检索（增强后查询）。"""
    return retrieve(question, top_c, top_d, top_k)


def flat_retrieve(question: str, top_k: int = 3) -> list[dict]:
    """扁平检索基线：全库文本块直接 top-k（无层次结构）。"""
    from src.embed.embed import embed_queries
    kb = load_kb()
    q = embed_queries([question])[0]
    scores = kb["chunk_vectors"] @ q
    hits = _topk(scores, top_k)
    return [{"chunk_id": kb["index"]["chunks"][i]["chunk_id"], "doc_id": kb["index"]["chunks"][i]["doc_id"],
             "score": round(s, 4), "text": kb["index"]["chunks"][i]["text"]} for i, s in hits]


def _print_results(title: str, results: list[dict]):
    print(f"--- {title} ---")
    for r in results:
        print(f"  [score={r['score']:.3f} | {r['doc_id']}]")
        print(f"    {r['text'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="C1 输入增强 + C2 DC-RAG 检索链路")
    parser.add_argument("question", help="用户查询")
    parser.add_argument("--top-c", type=int, default=2)
    parser.add_argument("--top-d", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print(f"原始问题: {args.question}")
    enhanced = enhance_query(args.question)
    print(f"C1 增强后: {enhanced}")
    print("=" * 60)

    print("\n>> DC-RAG 三级检索（类别→文档→文本块）")
    dc = dc_rag_retrieve(enhanced, args.top_c, args.top_d, args.top_k)
    _print_results(f"DC-RAG top{args.top_k}", dc)

    print("\n>> 扁平检索基线（全库直接 top-k）")
    flat = flat_retrieve(enhanced, args.top_k)
    _print_results(f"扁平 top{args.top_k}", flat)
    print("=" * 60)


if __name__ == "__main__":
    main()
