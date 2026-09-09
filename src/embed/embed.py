"""嵌入模型调用接口（模块 B1）：中文文本向量化。

选型说明：
  - 默认模型 BAAI/bge-small-zh-v1.5：中文语义检索效果优秀，512 维，约 100MB，
    适合本机 CPU 推理与 160 篇政务语料的规模。
  - 备选（修改 config.json 的 model_name/dim 即可切换）：
      BAAI/bge-base-zh-v1.5     768 维，效果更优，资源占用更高
      BAAI/bge-large-zh-v1.5    1024 维，效果最优，需更强算力
      BAAI/bge-m3               1024 维，多语言/多粒度

用法：
  python -m src.embed.embed --test        # 自测：对示例中文向量化并打印维度
  python -m src.embed.embed --text "深圳市政府工作报告"  # 单条文本向量化

说明：
  - BGE 系列对检索任务建议对 query 添加指令前缀 "为这个句子生成表示以用于检索相关文章："
    （本项目 query 增强在模块 C 实现，此处文档向量化直接编码）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

_model = None
_config = None


def load_config() -> dict:
    global _config
    if _config is None:
        with CONFIG_FILE.open(encoding="utf-8") as fp:
            _config = json.load(fp)
    return _config


def load_model():
    """懒加载 sentence-transformers 模型（首次调用时加载）。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        cfg = load_config()
        cache = PROJECT_ROOT / cfg["cache_dir"]
        cache.mkdir(parents=True, exist_ok=True)
        model_cache_name = "models--" + cfg["model_name"].replace("/", "--")
        snapshots_dir = cache / model_cache_name / "snapshots"
        local_snapshots = sorted(
            (p for p in snapshots_dir.glob("*") if (p / "config.json").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if snapshots_dir.exists() else []
        model_source = str(local_snapshots[0]) if local_snapshots else cfg["model_name"]
        offline = bool(local_snapshots)
        print(f"加载模型 {cfg['model_name']}（{'本地离线缓存' if offline else '首次下载'}: {model_source}）...")
        _model = SentenceTransformer(
            model_source,
            device=cfg["device"],
            cache_folder=str(cache),
            local_files_only=offline,
        )
    return _model


# BGE 系列检索建议：查询侧加指令前缀以提升检索效果（文档侧无需）
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def embed_texts(texts: list[str]) -> np.ndarray:
    """批量向量化，返回 (n, dim) 的 float32 数组，默认归一化。"""
    model = load_model()
    cfg = load_config()
    vectors = model.encode(texts, batch_size=cfg["batch_size"], normalize_embeddings=cfg["normalize"])
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(texts: list[str]) -> np.ndarray:
    """查询向量化：自动加 BGE 检索指令前缀（与文档侧区分）。"""
    return embed_texts([QUERY_PREFIX + t for t in texts])


def main():
    parser = argparse.ArgumentParser(description="嵌入模型调用接口")
    parser.add_argument("--test", action="store_true", help="自测")
    parser.add_argument("--text", help="单条文本向量化")
    args = parser.parse_args()

    cfg = load_config()
    if args.test:
        samples = [
            "2025年深圳市政府工作报告提出推动经济高质量发展。",
            "深圳加快建设全球领先的重要的先进制造业中心。",
            "深化粤港澳大湾区建设，推进前海开发开放。",
        ]
        vec = embed_texts(samples)
        print(f"模型: {cfg['model_name']}")
        print(f"向量形状: {vec.shape}（{len(samples)} 条 x {vec.shape[1]} 维）")
        # 相似度自检：同主题文本应比不同主题更相似
        cos = np.dot(vec, vec.T)
        sim_same = cos[0][1]
        sim_diff = cos[0][2]
        print(f"示例相似度: [0,1]={sim_same:.3f}, [0,2]={sim_diff:.3f}")
        if sim_same > sim_diff:
            print("相似度自检通过 [PASS]")
        else:
            print("相似度自检未通过（结果可接受，仅提示）")
        return
    if args.text:
        vec = embed_texts([args.text])
        print(f"维度: {vec.shape[1]}")
        print(f"向量(前16维): {vec[0][:16].round(4).tolist()}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
