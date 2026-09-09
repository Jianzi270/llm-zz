"""B6 知识库数据集导出：将层次化知识库导出为标准 JSONL 备份文件。

两种格式：
  - 文档格式（--format documents）：每行 {"title": 文档名, "content": 文档全文}。
  - 检索可复用格式（--format chunks）：从 data/processed/chunks.jsonl 复制为备份。

用法：
  python scripts/export_dataset.py                    # 导出文档 JSONL 数据集
  python scripts/export_dataset.py --format chunks    # 导出文本块备份

输出：
  data/export/documents.jsonl（160 篇文档全文）
  data/export/chunks_backup.jsonl（15546 个文本块）
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "export"


def export_documents():
    """按文档聚合文本块，导出通用文档 JSONL（title/content）。"""
    docs = defaultdict(list)
    for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        docs[c["doc_id"]].append(c["text"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "documents.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as fp:
        for doc_id in sorted(docs):
            fp.write(json.dumps({"title": doc_id, "content": "\n".join(docs[doc_id])}, ensure_ascii=False) + "\n")
    print(f"导出文档数据集: {len(docs)} 篇文档 -> {out}")


def export_chunks():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "chunks_backup.jsonl"
    out.write_bytes(CHUNKS_FILE.read_bytes())
    n = sum(1 for l in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"导出文本块备份: {n} 块 -> {out}")


def main():
    parser = argparse.ArgumentParser(description="知识库数据集导出")
    parser.add_argument("--format", choices=["documents", "chunks"], default="documents")
    args = parser.parse_args()
    export_documents() if args.format == "documents" else export_chunks()


if __name__ == "__main__":
    main()
