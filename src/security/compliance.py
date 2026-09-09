"""D5 安全与合规：敏感词过滤（输出可控）+ 审计日志（可审计）。

功能：
  - check_content(text)   检测文本命中敏感词，返回命中列表（空列表 = 通过）。
                          内置违法内容红线词表；可另建 data/config/sensitive_words.txt
                          追加自定义词表（每行一词，支持 # 注释），用于政治表述等按口径维护。
  - audit_log(entry)      追加一条审计记录到 data/logs/audit.jsonl（时间/问题/增强/来源/状态）。

用法：
  python -m src.security.compliance --check "测试文本"
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 内置敏感词表：明确的违法/红线内容（通用、稳定），政治类口径词由自定义词表按需维护
BUILTIN_SENSITIVE_WORDS = [
    "制毒", "贩毒", "走私枪支", "买卖枪支", "传播淫秽", "组织卖淫",
    "赌博网站", "诈骗教程", "黑客攻击教程", "攻击国家计算机信息系统",
]

# 自定义词表路径（可选，每行一个词，支持空行与 # 注释）
CUSTOM_WORDS_FILE = PROJECT_ROOT / "data" / "config" / "sensitive_words.txt"

LOG_DIR = PROJECT_ROOT / "data" / "logs"
AUDIT_FILE = LOG_DIR / "audit.jsonl"


def _load_words() -> list[str]:
    words = list(BUILTIN_SENSITIVE_WORDS)
    if CUSTOM_WORDS_FILE.exists():
        for line in CUSTOM_WORDS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    return words


_SENSITIVE = _load_words()


def check_content(text: str) -> list[str]:
    """检测文本中命中的敏感词，返回命中列表（空列表 = 合规通过）。"""
    if not text:
        return []
    return [w for w in _SENSITIVE if w and w in text]


def audit_log(entry: dict) -> None:
    """追加一条审计记录。entry 中的 ts 缺省取当前时间。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(entry)
    # 默认只保存问题指纹和长度；确有合规需要时可显式开启原文审计。
    if os.getenv("AUDIT_STORE_RAW_QUESTION", "0") != "1":
        for field in ("question", "enhanced", "answer_excerpt"):
            value = record.pop(field, None)
            if value:
                record[f"{field}_sha256"] = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
                record[f"{field}_chars"] = len(str(value))
    record.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    with AUDIT_FILE.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="安全合规工具")
    parser.add_argument("--check", help="检测文本是否命中敏感词")
    args = parser.parse_args()
    if args.check:
        hits = check_content(args.check)
        if hits:
            print(f"命中敏感词：{hits}")
            sys.exit(1)
        print("合规通过，未命中敏感词")


if __name__ == "__main__":
    main()
