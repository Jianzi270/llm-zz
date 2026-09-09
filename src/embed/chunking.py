"""文档切分模块（模块 B2）：将清洗后文档切分为文本块（chunk）。

策略：
  1. 段落 = 以空行分隔的自然段落（清洗后每段一行）
  2. 段落长度 > max_chunk_chars：按句子边界（。！？；）切分，避免截断语义
  3. 段落长度 < min_chunk_chars 且非标题：并入前一块，减少碎片
  4. 标题行（"一、xxx" / "（一）xxx" / "1.xxx"）保留为独立块，利于检索命中
  5. 每个块携带文档元数据（级别/区域/年份/标题），供 DC-RAG 使用
  6. 疑似无正文（残缺）文件自动跳过，避免导航噪声入库

用法：
  python -m src.embed.chunking [--max-chars 512] [--min-chars 30]

输出：
  data/processed/chunks.jsonl（每行一个块）+ 切分统计打印
"""
import argparse
import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEANED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

# 正文特征词（用于跳过残缺文件）
BODY_KEYWORDS = ("各位代表", "现在，我代表", "过去一年", "工作回顾", "请予审议", "报告如下")

# 标题行模式：一级"一、" / 二级"（一）" / 数字"1."/"1、" / "（1）"
HEADING_PATTERN = re.compile(
    r"^([一二三四五六七八九十百]+[、．.]|（[一二三四五六七八九十百]+）|[0-9]+[、.．]|（[0-9]+）)"
)
# 句子边界
SENTENCE_END = re.compile(r"[^。！？；\n]+[。！？；]?")


def is_heading(line: str) -> bool:
    return bool(HEADING_PATTERN.match(line.strip()))


def split_long_paragraph(par: str, max_chars: int) -> list[str]:
    """将超长段落按句切分，贪心合并句子使每块 <= max_chars。"""
    sentences = [s.strip() for s in SENTENCE_END.findall(par) if s.strip()]
    if not sentences:
        sentences = [par[i : i + max_chars] for i in range(0, len(par), max_chars)]
    else:
        sentences = [piece for sent in sentences
                     for piece in ([sent] if len(sent) <= max_chars else
                                   [sent[i:i + max_chars] for i in range(0, len(sent), max_chars)])]
    chunks, buf = [], ""
    for sent in sentences:
        if buf and len(buf) + len(sent) > max_chars:
            chunks.append(buf)
            buf = sent
        else:
            buf = buf + sent if buf else sent
    if buf:
        chunks.append(buf)
    return chunks


def split_document(doc_id: str, text: str, meta: dict, max_chars: int, min_chars: int) -> list[dict]:
    """将单篇文档切分为块列表（保留元数据）。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks: list[dict] = []
    pending_short = ""  # 待并入前块的短段落

    for par in paragraphs:
        if len(par) < min_chars and not is_heading(par):
            pending_short = pending_short + par if pending_short else par
            continue
        # 先合并挂起的短段落
        if pending_short:
            if chunks and len(chunks[-1]["text"]) + len(pending_short) <= max_chars:
                chunks[-1]["text"] += pending_short
            else:
                par = pending_short + par
            pending_short = ""
        if len(par) <= max_chars:
            parts = [par]
        else:
            parts = split_long_paragraph(par, max_chars)
        for part in parts:
            chunk = {
                "doc_id": doc_id,
                "doc_title": meta.get("title", ""),
                "source_level": meta.get("source_level", ""),
                "region": meta.get("region", ""),
                "year": meta.get("report_year", ""),
                "text": part,
            }
            chunks.append(chunk)
    if pending_short:
        if chunks and len(chunks[-1]["text"]) + len(pending_short) <= max_chars:
            chunks[-1]["text"] += pending_short
        else:
            for part in split_long_paragraph(pending_short, max_chars):
                chunks.append({
                    "doc_id": doc_id,
                    "doc_title": meta.get("title", ""),
                    "source_level": meta.get("source_level", ""),
                    "region": meta.get("region", ""),
                    "year": meta.get("report_year", ""),
                    "text": part,
                })
    # 附加全局块编号
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{Path(doc_id).stem}_{i}"
    return chunks


def main():
    parser = argparse.ArgumentParser(description="文档切分")
    parser.add_argument("--max-chars", type=int, default=512, help="单块最大字符数")
    parser.add_argument("--min-chars", type=int, default=30, help="短段落并入阈值")
    args = parser.parse_args()

    meta_map = {}
    if METADATA_FILE.exists():
        with METADATA_FILE.open(encoding="utf-8-sig", newline="") as fp:
            for row in csv.DictReader(fp):
                meta_map[row["filename"]] = row

    all_chunks = []
    skipped = []
    for f in sorted(CLEANED_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        if not any(k in text for k in BODY_KEYWORDS):
            skipped.append(f.name)
            continue
        meta = meta_map.get(f.name, {})
        all_chunks.extend(split_document(f.name, text, meta, args.max_chars, args.min_chars))

    with OUT_FILE.open("w", encoding="utf-8", newline="\n") as fp:
        for c in all_chunks:
            fp.write(json.dumps(c, ensure_ascii=False) + "\n")

    from collections import Counter
    lens = [len(c["text"]) for c in all_chunks]
    print(f"文档数: {len(list(CLEANED_DIR.glob('*.txt')))}, 切分块数: {len(all_chunks)}")
    print(f"跳过残缺文档: {len(skipped)} -> {skipped}")
    print(f"块长 min/median/max: {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    print(f"级别分布: {dict(Counter(c['source_level'] for c in all_chunks))}")
    print(f"输出: {OUT_FILE}")


if __name__ == "__main__":
    main()
