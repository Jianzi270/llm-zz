"""一键爬取 + 清洗流水线（手动执行接口，不做自动调度、不执行 git 操作）。

功能：手动一键串联爬取与清洗，替代人工分步执行。
  1. 爬取：调用 scripts/crawler/crawler.py 增量抓取新政策报告 -> data/raw/crawled/
  2. 清洗：调用 scripts/clean_text.py 对 data/raw/crawled/ 清洗 -> data/processed/cleaned/
  3. 报告：汇总本次新增文件、清洗结果与残缺文件提醒（供人工审核）

安全说明：
  - 本脚本不执行任何 git 操作（不自动提交/推送）；运行后请人工审核变更，
    确认无误后再手动执行 git add / commit / push。
  - 全部步骤幂等（已存在文件自动跳过），可重复运行。

命令行用法：
  python scripts/auto_update.py               # 一键：爬取 + 清洗 + 报告
  python scripts/auto_update.py --dry-run     # 仅预览候选（爬虫 --dry-run），不下载不清洗
  python scripts/auto_update.py --skip-crawl  # 跳过爬取，仅清洗 data/raw/crawled/ 中新增文件

代码接口用法：
  from scripts.auto_update import run_pipeline
  run_pipeline()               # 一键爬取 + 清洗
  run_pipeline(dry_run=True)   # 预览
  run_pipeline(skip_crawl=True)  # 仅清洗
"""
import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CRAWLED_DIR = PROJECT_ROOT / "data" / "raw" / "crawled"      # 爬虫原始输出
CLEANED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"  # 清洗后语料
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaning_report.csv"


def run(cmd: list[str]) -> int:
    """运行子命令并透传输出，返回退出码。"""
    print(f"\n>> {' '.join(cmd)}")
    r = subprocess.run([sys.executable, *cmd], cwd=str(PROJECT_ROOT))
    return r.returncode


def cleaned_filenames() -> set[str]:
    return {f.name for f in CLEANED_DIR.glob("*.txt")} if CLEANED_DIR.exists() else set()


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


def summarize(dry_run: bool, skip_crawl: bool) -> int:
    """输出审核摘要：本次新增 / 残缺提醒。"""
    print("\n" + "=" * 66)
    print("一键爬取 + 清洗 审核摘要")
    print("=" * 66)

    if dry_run:
        print("本次为 --dry-run 预览模式：未下载、未清洗，请人工确认候选后正式运行。")
        return 0

    flags = read_report_flags()
    suspicious = [f for f, fl in flags.items() if fl["suspicious_empty_body"]]
    nav = [f for f, fl in flags.items() if fl["nav_residue"]]
    suspicious = [f for f in suspicious if f not in nav]  # 导航残留同时标记的归入 nav

    print(f"清洗后语料总数：{len(cleaned_filenames())} 篇（目录 {CLEANED_DIR}）")
    print(f"清洗报告：{REPORT_FILE}")

    changed = []
    for f in sorted(CLEANED_DIR.glob("*.txt")):
        raw = CRAWLED_DIR / f.name
        if raw.exists():
            changed.append(f.name)
    print(f"本次涉及（来自爬虫目录）文件数：{len(changed)}")
    if changed:
        print("  " + "\n  ".join(changed))
    else:
        print("  无新文件（爬虫未发现新增，或全部已在库中）")

    if suspicious or nav:
        print("\n⚠ 需要人工审核的文件：")
        for f in suspicious:
            print(f"  [疑似无正文/残缺] {f}（仅爬取了网页导航，建议检查后删除或补抓）")
        for f in nav:
            print(f"  [含导航残留] {f}（已记录未删除，建议人工检查正文完整性）")
    else:
        print("\n无残缺/异常文件提醒。")

    print("\n说明：本脚本未执行任何 git 操作。请人工审核上述变更，确认无误后手动提交：")
    print("  git add <变更文件> && git commit -m \"data: 新增 XX 篇报告\" && git push")
    return 0


def run_pipeline(dry_run: bool = False, skip_crawl: bool = False) -> int:
    """一键爬取 + 清洗流水线（可编程接口）。

    Args:
        dry_run: 仅预览爬虫候选，不下载不清洗。
        skip_crawl: 跳过爬取，仅清洗 data/raw/crawled/ 中新增文件。

    Returns:
        0 成功；1 失败（爬取或清洗异常时中止）。
    """
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 一键流水线开始"
          + ("（预览模式）" if dry_run else ""))

    # 步骤 1：增量爬取（已存在文件自动跳过）
    if skip_crawl:
        print("\n[跳过爬取] --skip-crawl")
    else:
        crawl_args = ["-m", "scripts.crawler.crawler"]
        if dry_run:
            crawl_args.append("--dry-run")
        if run(crawl_args) != 0:
            print("爬取步骤失败，已中止（不执行清洗）。")
            return 1

    # 步骤 2：清洗爬虫目录（幂等：仅输出新增/变化文件；报告按文件名合并去重）
    if not dry_run:
        if run(["scripts/clean_text.py", str(CRAWLED_DIR)]) != 0:
            print("清洗步骤失败。")
            return 1

    return summarize(dry_run, skip_crawl)


def main():
    parser = argparse.ArgumentParser(description="一键爬取 + 清洗流水线（手动执行，不执行 git 操作）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览候选，不下载不清洗")
    parser.add_argument("--skip-crawl", action="store_true", help="跳过爬取，仅清洗新增文件")
    args = parser.parse_args()
    sys.exit(run_pipeline(dry_run=args.dry_run, skip_crawl=args.skip_crawl))


if __name__ == "__main__":
    main()
