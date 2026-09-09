"""E1 系统集成与联调：验证 数据层 → 知识库 → 检索 → 生成 → 应用 全链路无断点。

用法：
  python scripts/e2e_check.py               # 快速联调（数据/知识库/检索/应用健康检查，不调 LLM）
  python scripts/e2e_check.py --with-llm    # 完整联调（含 C1 增强 + C4 生成 + 应用问答，会调用 LLM）

输出：
  data/eval/e2e_report.json（逐项检查结果 + 汇总）
  全部通过返回退出码 0，任一项失败返回 1。
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLES = [
    "深圳2025年GDP增长目标是多少？",
    "罗湖区城市更新成效如何？",
]
SAMPLES_FAST = SAMPLES[:1]
EXPECTED_DOC = "深圳市政府工作报告_2025年.txt"
EXPECTED_FACT = "5.5"

checks: list[dict] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def check_data_layer() -> None:
    """数据层：清洗语料 / 元数据 / 文本块 / 摘要。"""
    cleaned_dir = PROJECT_ROOT / "data" / "processed" / "cleaned"
    n_txt = len(list(cleaned_dir.glob("*.txt")))
    record("数据层-清洗语料", n_txt >= 160, f"清洗语料 {n_txt} 篇")

    meta = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"
    n_meta = 0
    if meta.exists():
        with meta.open(encoding="utf-8") as fp:
            n_meta = sum(1 for _ in csv.DictReader(fp))
    record("数据层-元数据", n_meta == n_txt, f"元数据 {n_meta} 条 / 语料 {n_txt} 篇")

    chunks = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    n_chunks = sum(1 for l in chunks.read_text(encoding="utf-8").splitlines() if l.strip()) if chunks.exists() else 0
    record("数据层-文本块", n_chunks >= 15000, f"文本块 {n_chunks}")

    sums = PROJECT_ROOT / "data" / "processed" / "summaries.jsonl"
    n_sums = sum(1 for l in sums.read_text(encoding="utf-8").splitlines() if l.strip()) if sums.exists() else 0
    record("数据层-摘要", n_sums == n_txt, f"摘要 {n_sums} 篇 / 语料 {n_txt} 篇")


def check_kb() -> None:
    """知识库：向量文件与索引（8 类 / 159 篇）。"""
    kb_dir = PROJECT_ROOT / "data" / "knowledge_base"
    needed = ["chunk_vectors.npy", "doc_vectors.npy", "category_vectors.npy", "index.json", "manifest.json"]
    missing = [f for f in needed if not (kb_dir / f).exists()]
    record("知识库-向量文件", not missing, "缺失: " + ", ".join(missing) if missing else "5 个文件齐全")

    if (kb_dir / "index.json").exists():
        idx = json.loads((kb_dir / "index.json").read_text(encoding="utf-8"))
        n_clusters = len(idx.get("cluster_docs", {}))
        n_docs = len(idx.get("doc_ids", []))
        has_metadata = len(idx.get("doc_metadata", {})) == n_docs
        record("知识库-索引结构", n_clusters == 8 and n_docs >= 160 and has_metadata,
               f"{n_clusters} 类 / {n_docs} 篇文档")


def check_retrieval() -> None:
    """检索：DC-RAG 三级检索与扁平基线均可返回结果（无 LLM 调用）。"""
    from src.retrieval.query_pipeline import dc_rag_retrieve, flat_retrieve
    q = "深圳2025年经济社会发展目标"
    dc = dc_rag_retrieve(q, 2, 3, 3)
    dc_docs = {item["doc_id"] for item in dc}
    record("检索-DC-RAG", EXPECTED_DOC in dc_docs,
           f"返回 {len(dc)} 块，目标文档={'命中' if EXPECTED_DOC in dc_docs else '未命中'}")
    flat = flat_retrieve(q, 3)
    record("检索-扁平基线", len(flat) > 0, f"返回 {len(flat)} 块")


def check_generate() -> None:
    """生成：C1 增强 + C4 生成（调用 LLM）。"""
    from src.generate.answer import generate_answer
    for q in SAMPLES_FAST:
        t0 = time.time()
        try:
            r = generate_answer(q, top_c=2, top_d=3, top_k=3)
            source_docs = {s["doc_id"] for s in r["sources"]}
            ok = bool(r["answer"]) and EXPECTED_DOC in source_docs and EXPECTED_FACT in r["answer"]
            record("生成-" + q[:18], ok,
                   f"来源 {len(r['sources'])} 篇 / 答案 {len(r['answer'])} 字 / {round(time.time()-t0, 1)}s")
        except Exception as e:
            record("生成-" + q[:18], False, f"异常: {e}")


def check_app(with_llm: bool) -> None:
    """应用：健康检查 + 可选问答接口。"""
    base = "http://127.0.0.1:8000"
    try:
        with urllib.request.urlopen(base + "/api/health", timeout=10) as resp:
            body = json.loads(resp.read().decode())
            ok = resp.status == 200 and body.get("status") == "ready"
        record("应用-健康检查", ok)
    except Exception as e:
        record("应用-健康检查", False, f"服务未启动或异常: {e}")
        return
    if with_llm:
        for q in SAMPLES_FAST:
            try:
                req = urllib.request.Request(
                    base + "/api/ask", data=json.dumps({"question": q}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    body = json.loads(resp.read().decode())
                source_docs = {s["doc_id"] for s in body.get("sources", [])}
                answer_ok = EXPECTED_DOC in source_docs and EXPECTED_FACT in body.get("answer", "")
                record("应用-问答-" + q[:12], answer_ok,
                       f"答案 {len(body.get('answer', ''))} 字")
            except Exception as e:
                record("应用-问答-" + q[:12], False, f"异常: {e}")


def main():
    parser = argparse.ArgumentParser(description="E1 系统集成与联调")
    parser.add_argument("--with-llm", action="store_true", help="包含生成链路与应用问答（调用 LLM）")
    args = parser.parse_args()
    checks.clear()

    print("=" * 66)
    print("E1 系统集成与联调" + ("（完整模式，含 LLM）" if args.with_llm else "（快速模式，无 LLM）"))
    print("=" * 66)
    t0 = time.time()
    check_data_layer()
    check_kb()
    check_retrieval()
    if args.with_llm:
        check_generate()
    check_app(args.with_llm)
    print("-" * 66)

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    report = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "full" if args.with_llm else "quick",
        "passed": passed, "total": total,
        "cost_s": round(time.time() - t0, 2),
        "checks": checks,
    }
    out = PROJECT_ROOT / "data" / "eval" / "e2e_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"联调结果：{passed}/{total} 通过（{report['cost_s']}s），报告 -> {out}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
