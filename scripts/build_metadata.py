"""元数据标注脚本（模块 A2）：从文件名解析文档元数据，输出 metadata.csv。

用法：python scripts/build_metadata.py
输出：data/metadata/metadata.csv（UTF-8 with BOM）
字段：filename, title, source_level, region, report_year
说明：解析规则基于文件名约定，生成后需人工抽查复核（尤其 region/source_level 为"未知"的行）。
"""
import csv
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 元数据基于清洗后的最终语料目录生成（含爬虫新增语料）
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"
OUT_DIR = PROJECT_ROOT / "data" / "metadata"
OUT_FILE = OUT_DIR / "metadata.csv"

# 行政区名（含历史名称），按更具体名称优先匹配
DISTRICTS = ["大鹏新区", "南山区", "福田区", "罗湖区", "龙岗区", "宝安区",
             "龙华区", "光明新区", "光明区", "坪山新区", "坪山区", "盐田区",
             "深汕特别合作区"]
# 历史区名统一为现区名
REGION_NORMALIZE = {"光明新区": "光明区", "坪山新区": "坪山区"}


def parse(filename: str) -> dict:
    stem = filename[:-4] if filename.endswith(".txt") else filename
    # 报告内容年份：取文件名中第一个 20xx（如"大鹏新区管委会2014年工作报告_2015.txt"→2014）
    years = re.findall(r"20\d{2}", stem)
    report_year = years[0] if years else ""

    # 行政区
    region = ""
    for d in DISTRICTS:
        if d in stem:
            region = REGION_NORMALIZE.get(d, d)
            break

    # 来源级别与区域兜底（国家级/省级/市级文件的 region）
    if "国务院" in stem:
        level = "国家级"
        region = region or "全国"
    elif "广东省" in stem:
        level = "省级"
        region = region or "广东省"
    elif region:
        level = "区级"
    elif "深圳市" in stem:
        level = "市级"
        region = region or "深圳市"
    else:
        level = "未知"

    # 标题：去掉文件末尾的 _年份 后缀
    title = re.sub(r"_20\d{2}$", "", stem)

    return {
        "filename": filename,
        "title": title,
        "source_level": level,
        "region": region if region else "未知",
        "report_year": report_year,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [parse(f.name) for f in sorted(DATA_DIR.glob("*.txt"))]
    with OUT_FILE.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["filename", "title", "source_level", "region", "report_year"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"共 {len(rows)} 个文件")
    print("级别分布:", dict(Counter(r["source_level"] for r in rows)))
    print("区域分布:", dict(Counter(r["region"] for r in rows)))
    unknown = [r["filename"] for r in rows if r["source_level"] == "未知" or r["region"] == "未知" or not r["report_year"]]
    if unknown:
        print("警告：以下文件解析可能有误，需人工复核：")
        for u in unknown:
            print("  -", u)
    print(f"输出: {OUT_FILE}")


if __name__ == "__main__":
    main()
