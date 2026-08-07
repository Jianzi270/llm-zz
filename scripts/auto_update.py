"""一键爬取 + 清洗 + 入知识库流水线（手动执行接口，不做自动调度、不执行 git 操作）。

功能：手动一键执行，将新政策数据完整送入知识库并可被检索。
  1. 爬取   ：调用 scripts/crawler/crawler.py 增量抓取 -> data/raw/crawled/
  2. 清洗   ：调用 scripts/clean_text.py -> data/processed/cleaned/
  3. 元数据 ：重建 metadata.csv（增量文件的解析）
  4. 摘要   ：增量生成（仅新文档；LLM 优先，失败提取式兜底）
  5. 切分   ：重建 chunks.jsonl（残缺文件自动跳过）
  6. 聚类   ：GMM 软聚类（--n-clusters 8，保持类别体系稳定）
  7. 知识库 ：重建 data/knowledge_base/（块向量缓存按数量校验复用）
  8. 验证   ：确认新文档已入知识库 + 检索冒烟测试

安全说明：
  - 不执行任何 git 操作（不自动提交/推送）；运行后请人工审核变更，
    确认无误后再手动执行 git add / commit / push。
  - 全部步骤幂等（已存在文件/摘要自动跳过），可重复运行。

命令行用法：
  python scripts/auto_update.py                 # 一键：爬取+清洗+入知识库
  python scripts/auto_update.py --dry-run       # 仅预览爬取候选（后续步骤不执行）
  python scripts/auto_update.py --skip-crawl    # 跳过爬取（仅清洗+知识库更新）
  python scripts/auto_update.py --skip-kb       # 仅爬取+清洗（不更新知识库）

代码接口用法：
  from scripts.auto_update import run_pipeline
  run_pipeline()                      # 一键全流程
  run_pipeline(dry_run=True)          # 预览
  run_pipeline(skip_crawl=True)       # 不爬取
  run_pipeline(skip_kb=True)          # 不更新知识库
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CRAWLED_DIR = PROJECT_ROOT / "data" / "raw" / "crawled"        # 爬虫原始输出
CLEANED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"  # 清洗后语料
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaning_report.csv"
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"

# 残缺文件判定：正文特征词（与 chunking 保持一致）
BODY_KEYWORDS = ("各位代表", "现在，我代表", "过去一年", "工作回顾", "请予审议", "报告如下")


def run(cmd: list[str]) -> int:
    """运行子命令并透传输出，返回退出码。"""
    print(f"\n>> {' '.join(cmd)}")
    r = subprocess.run([sys.executable, *cmd], cwd=str(PROJECT_ROOT))
    return r.returncode


def cleaned_filenames() -> set[str]:
    return {f.name for f in CLEANED_DIR.glob("*.txt")} if CLEANED_DIR.exists() else set()


def has_body(text: str) -> bool:
    return any(k in text for k in BODY_KEYWORDS)


def read_report_flags() -> dict[str, dict]:
    """读取清洗报告，返回 {filename: {suspicious_empty_body, nav_residue}}。"""
    flags: dict[str, dict] = {}
    if not REPORT_FILE.exists():
        return flags
    with REPORT_FILE.open(encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            flags[r["filename"]] = {
                "suspicious_empty_body": r.get("suspicious_empty_body") == "True",
                "nav_residue": r.get("nav_residue") == "True",
            }
    return flags


def summarize_clean(dry_run: bool) -> None:
    """输出爬取+清洗阶段的审核摘要。"""
    print("\n" + "=" * 66)
    print("爬取 + 清洗 审核摘要")
    print("=" * 66)
    if dry_run:
        print("本次为 --dry-run 预览模式：未下载、未清洗，请人工确认候选后正式运行。")
        return

    flags = read_report_flags()
    suspicious = [f for f, fl in flags.items() if fl["suspicious_empty_body"]]
    nav = [f for f, fl in flags.items() if fl["nav_residue"]]
    suspicious = [f for f in suspicious if f not in nav]

    print(f"清洗后语料总数：{len(cleaned_filenames())} 篇（目录 {CLEANED_DIR}）")
    print(f"清洗报告：{REPORT_FILE}")

    changed = [f.name for f in sorted(CLEANED_DIR.glob("*.txt")) if (CRAWLED_DIR / f.name).exists()]
    print(f"本次涉及（来自爬虫目录）文件数：{len(changed)}")
    print("  " + "\n  ".join(changed) if changed else "  无新文件（爬虫未发现新增，或全部已在库中）")

    if suspicious or nav:
        print("\n⚠ 需要人工审核的文件：")
        for f in suspicious:
            print(f"  [疑似无正文/残缺] {f}（仅爬取了网页导航，建议检查后删除或补抓）")
        for f in nav:
            print(f"  [含导航残留] {f}（已记录未删除，建议人工检查正文完整性）")
    else:
        print("\n无残缺/异常文件提醒。")


def update_summaries() -> int:
    """增量摘要：仅对无摘要的新文档生成；LLM 优先，失败提取式兜底。"""
    import src.summarize.summarize as sm

    sm.load_env()
    cfg = sm.load_config()
    out_file = PROJECT_ROOT / cfg["output_file"]
    existing = sm.existing_summaries(out_file)
    docs = sorted(CLEANED_DIR.glob("*.txt"))
    new_docs = [f for f in docs if f.name not in existing]
    print(f"\n摘要：已有 {len(existing)} 篇，待生成 {len(new_docs)} 篇")
    if not new_docs:
        print("无新摘要需要生成。")
        return 0

    key, _, _ = sm.get_llm_settings(cfg)
    results = dict(existing)
    fallback_used = []
    for i, f in enumerate(new_docs, 1):
        text = f.read_text(encoding="utf-8")
        summary = ""
        if key:
            try:
                summary = sm.call_llm(cfg, Path(f.name).stem, text)
            except Exception as e:
                fallback_used.append(f"{f.name}: {str(e)[:60]}")
        if not summary:  # 无 Key 或 LLM 失败 → 提取式兜底，保证新文档可入知识库
            summary = sm.extractive_summary(text, cfg["extractive"]["num_sentences"])
            if f.name not in [fb.split(":")[0] for fb in fallback_used]:
                fallback_used.append(f"{f.name}: 未配置 API Key 或无摘要，使用提取式兜底")
        results[f.name] = json.dumps({"doc_id": f.name, "summary": summary}, ensure_ascii=False)
        print(f"  [{i}/{len(new_docs)}] {f.name}（{len(summary)} 字）")
        time.sleep(0.3)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8", newline="\n") as fp:
        for doc_id in sorted(results):
            fp.write(results[doc_id] + "\n")
    if fallback_used:
        print(f"⚠ 以下文档使用提取式兜底（如需 LLM 摘要可稍后重跑本脚本）：")
        for fb in fallback_used:
            print(f"  - {fb}")
    print(f"完成：{len(results)} 篇摘要 -> {out_file}")
    return 0


def verify_kb(new_docs: list[str]) -> bool:
    """验证新文档已入知识库 + 检索冒烟测试。"""
    print("\n" + "=" * 66)
    print("知识库更新验证")
    print("=" * 66)
    from src.retrieval.kb import load_kb

    kb = load_kb()
    idx = kb["index"]
    in_kb = set(idx["doc_ids"])
    missing = [d for d in new_docs if d not in in_kb]
    for d in new_docs:
        if (CLEANED_DIR / d).exists():
            ok = d in in_kb
            print(f"  [{'入库' if ok else '未入库'}] {d}"
                  + ("" if ok else "（残缺/无正文会被自动跳过，属正常）"))
    if missing:
        no_body = [d for d in missing if (CLEANED_DIR / d).exists() and not has_body((CLEANED_DIR / d).read_text(encoding="utf-8"))]
        if no_body:
            print(f"  其中 {len(no_body)} 篇为残缺文件（无正文），已按规则跳过：{no_body}")

    from src.retrieval.query_pipeline import dc_rag_retrieve
    q = "深圳2026年经济社会发展目标"
    r = dc_rag_retrieve(q, 2, 3, 3)
    ok = len(r) > 0
    print(f"  检索冒烟测试：'{q}' -> 返回 {len(r)} 块" + (f"，首篇 {r[0]['doc_id']}" if r else ""))
    return ok


def run_pipeline(dry_run: bool = False, skip_crawl: bool = False, skip_kb: bool = False) -> int:
    """一键：爬取 + 清洗 + 入知识库（可编程接口）。

    Args:
        dry_run: 仅预览爬虫候选，不下载不清洗不更新知识库。
        skip_crawl: 跳过爬取。
        skip_kb: 跳过知识库更新（仅爬取+清洗）。

    Returns:
        0 成功；1 失败。
    """
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 一键流水线开始"
          + ("（预览模式）" if dry_run else ""))
    before = cleaned_filenames()

    # 步骤 1：增量爬取（已存在文件自动跳过）
    if skip_crawl:
        print("\n[跳过爬取] --skip-crawl")
    else:
        crawl_args = ["-m", "scripts.crawler.crawler"]
        if dry_run:
            crawl_args.append("--dry-run")
        if run(crawl_args) != 0:
            print("爬取步骤失败，已中止。")
            return 1

    # 步骤 2：清洗爬虫目录（幂等；报告按文件名合并去重）
    if not dry_run:
        if run(["scripts/clean_text.py", str(CRAWLED_DIR)]) != 0:
            print("清洗步骤失败。")
            return 1

    summarize_clean(dry_run)

    # 步骤 3-8：知识库更新（dry-run 不执行）
    if dry_run or skip_kb:
        if skip_kb:
            print("\n[跳过知识库更新] --skip-kb（新数据暂未入库，可稍后运行本脚本补全）")
        return 0

    after = cleaned_filenames()
    new_docs = sorted(after - before)
    if not new_docs:
        print("\n无新增语料，仍需重建索引以保持一致，继续知识库更新...")
    else:
        print(f"\n新增语料 {len(new_docs)} 篇：{'、'.join(new_docs)}")

    # 摘要（增量，放前面，供聚类使用）
    if update_summaries() != 0:
        return 1
    # 元数据 → 切分 → 聚类 → 知识库
    for name, cmd in [
        ("元数据", ["scripts/build_metadata.py"]),
        ("切分", ["-m", "src.embed.chunking"]),
        ("聚类", ["-m", "src.cluster.gmm_cluster", "--n-clusters", "8"]),
        ("知识库", ["-m", "src.retrieval.build_kb"]),
    ]:
        if run(cmd) != 0:
            print(f"[{name}] 步骤失败，已中止。")
            return 1

    verify_kb(new_docs)

    print("\n" + "=" * 66)
    print("本流水线未执行任何 git 操作。请人工审核后手动提交：")
    print("  git add <变更文件> && git commit -m \"data: 新增 XX 篇报告并入知识库\" && git push")
    return 0


def main():
    parser = argparse.ArgumentParser(description="一键爬取+清洗+入知识库（手动执行，不执行 git 操作）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览候选，不下载不清洗")
    parser.add_argument("--skip-crawl", action="store_true", help="跳过爬取，仅清洗+更新知识库")
    parser.add_argument("--skip-kb", action="store_true", help="仅爬取+清洗，不更新知识库")
    args = parser.parse_args()
    sys.exit(run_pipeline(dry_run=args.dry_run, skip_crawl=args.skip_crawl, skip_kb=args.skip_kb))


if __name__ == "__main__":
    main()
