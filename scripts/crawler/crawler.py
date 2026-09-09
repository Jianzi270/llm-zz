"""政策数据爬虫（模块 A4）：动态抓取政府工作报告，支持增量更新与定时调度。

用法：
  python -m scripts.crawler.crawler --dry-run    # 仅解析列表页，预览候选（不下载）
  python -m scripts.crawler.crawler              # 执行增量抓取（跳过已下载/已入库数据）
  python -m scripts.crawler.crawler --limit 2    # 本次最多下载 2 篇

增量规则：
  - 已存在于 data/raw/crawled/ 的文件跳过
  - 已存在于 data/processed/cleaned/ 的文件跳过（避免与存量语料重复）

定时调度（Windows 计划任务，每日 08:00）：
  schtasks /create /tn "zizheng_crawler" /tr "python D:\\Projects\\资政大模型\\scripts\\crawler\\crawler.py" /sc daily /st 08:00
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

# 本地存量语料目录（用于增量去重）
EXISTING_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "cleaned",
    PROJECT_ROOT / "data" / "raw" / "crawled",
]

# 报告正文起点特征（政府工作报告）
BODY_MARK = "各位代表"


def load_config() -> dict:
    with CONFIG_FILE.open(encoding="utf-8") as fp:
        return json.load(fp)


def fetch(url: str, cfg: dict) -> requests.Response:
    """带重试的请求；https 失败自动回退 http（部分政务站点 TLS 兼容问题）。"""
    headers = {"User-Agent": cfg["request"]["user_agent"]}
    timeout = cfg["request"]["timeout"]
    retries = cfg["request"]["max_retries"]
    candidates = [url]
    if url.startswith("https://"):
        candidates.append("http://" + url[len("https://"):])
    last_err: requests.RequestException | None = None
    for _ in range(retries + 1):
        for u in candidates:
            try:
                return requests.get(u, timeout=timeout, headers=headers)
            except requests.RequestException as e:
                last_err = e
        time.sleep(1)
    raise last_err if last_err else RuntimeError(f"请求失败: {url}")


def parse_list(html: str, cfg_src: dict, base_url: str = "") -> list[dict]:
    """解析栏目列表页，返回 [{title, year, url}]。"""
    soup = BeautifulSoup(html, "html.parser")
    items = []
    seen = set()
    pattern = cfg_src["link_pattern"]
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if pattern not in href or href in seen:
            continue
        text = a.get_text(strip=True)
        if not text:
            continue
        year_match = re.search(r"(20\d{2})\s*年", text) or re.search(r"/(20\d{2})/", href)
        if not year_match:
            continue
        year = year_match.group(1)
        seen.add(href)
        items.append({"title": text, "year": year, "url": urljoin(base_url, href)})
    return items


def extract_body(html: str, selector: str) -> str:
    """从详情页提取报告正文：定位正文容器，自'各位代表'起截取。"""
    soup = BeautifulSoup(html, "html.parser")
    el = soup.select_one(selector)
    fallback = soup.body or soup
    text = el.get_text("\n", strip=True) if el else fallback.get_text("\n", strip=True)
    idx = text.find(BODY_MARK)
    return text[idx:] if idx >= 0 else text


def existing_filenames() -> set[str]:
    """存量语料文件名集合（用于增量去重）。"""
    names = set()
    for d in EXISTING_DIRS:
        if d.exists():
            names.update(f.name for f in d.glob("*.txt"))
    return names


def main():
    parser = argparse.ArgumentParser(description="政策数据爬虫")
    parser.add_argument("--dry-run", action="store_true", help="仅预览候选，不下载")
    parser.add_argument("--limit", type=int, default=0, help="本次最大下载数（0=不限）")
    parser.add_argument("--source", help="仅运行指定来源名称")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = existing_filenames()

    total_new = 0
    failures = 0
    for src in cfg["sources"]:
        if args.source and src["name"] != args.source:
            continue
        print(f"== 来源: {src['name']} ==")
        if src.get("documents"):
            items = src["documents"]
        else:
            resp = fetch(src["list_url"], cfg)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            items = parse_list(resp.text, src, src["list_url"])
        print(f"列表页候选 {len(items)} 条")
        for it in items:
            fname = it.get("filename") or f"{src['title_prefix']}_{it['year']}年.txt"
            if fname in existing:
                print(f"  [跳过] 已有 {fname}")
                continue
            print(f"  [新增] {fname}  <-  {it['url']}")
            if args.dry_run:
                continue
            try:
                detail = fetch(it["url"], cfg)
                detail.raise_for_status()
                detail.encoding = detail.apparent_encoding
                body = extract_body(detail.text, src["content_selector"])
                if len(body) < 500:
                    print(f"  [失败] 正文过短（{len(body)}字符），未保存")
                    failures += 1
                    continue
                (out_dir / fname).write_text(body + "\n", encoding="utf-8", newline="\n")
                print(f"  [已保存] {len(body)} 字符")
                total_new += 1
            except requests.RequestException as e:
                print(f"  [失败] {e}")
                failures += 1
            if args.limit and total_new >= args.limit:
                print("达到 --limit 限制，停止")
                return
    print(f"完成，本次新增 {total_new} 篇，失败 {failures} 篇，输出目录: {out_dir}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
