# 资政大模型

基于垂直领域大模型的**政府智库报告生成智能体**（哈尔滨工业大学（深圳）大一年度项目）。

**结题目标**：搭建好知识库，部署大语言模型，实现政务文字及回答生成专业化、结构化。

**核心链路**：`输入增强与向量化 → 文本检索查询 → 大模型生成输出`

---

## 目录结构

```
资政大模型/
├── README.md                              # 项目说明（本文件）
├── .gitignore                             # 忽略规则
├── data/                                  # 数据
│   ├── gov_work_reports_sz_txt_cleaned/   # 原始清洗语料（146 篇，2014-2025）
│   ├── processed/cleaned/                 # 清洗后最终语料（160 篇，2001-2026）
│   ├── raw/crawled/                       # 爬虫抓取原始文本（14 篇）
│   ├── inventory/                         # 数据清单
│   ├── metadata/                          # 文档元数据
│   └── export/                            # 数据集导出备份（dify_dataset.jsonl 等）
├── src/                                   # 核心源码
│   ├── embed/                             # 文档切分与向量化（BGE 嵌入）
│   ├── summarize/                         # 文档摘要生成（LLM 双模式）
│   ├── cluster/                           # GMM 软聚类
│   ├── retrieval/                         # DC-RAG 检索与生成（三级检索）
│   └── generate/                          # 答案生成（C4 链路）
├── app/                                   # 自研智能体 Web 应用（替代 Dify 平台）
│   ├── server.py                          # Flask 服务（/api/ask 问答接口）
│   └── templates/index.html               # 聊天界面
├── scripts/                               # 爬虫与工具脚本
│   ├── crawler/                           # 政策数据爬虫
│   └── export_dify_dataset.py             # 知识库数据集导出（Dify 兼容格式 + 备份）
├── dify/                                  # Dify 配置（平台侧备选方案）
│   └── Rag.yml                            # Dify advanced-chat 应用配置（已备份）
└── docs/                                  # 项目文档
    ├── 资政大模型--大一年度项目立项报告 (1).pdf   # 立项报告
    ├── 资政大模型-项目要求.md                    # 项目要求（现状盘点/需求/验收标准）
    └── 资政大模型-结题任务清单.md                # 结题任务清单（重建导向）
```

> 注：`src/`、`scripts/` 下为规划中的子目录，随模块开发逐步落地。

---

## 相关文档

| 文档 | 说明 |
| --- | --- |
| [docs/资政大模型-项目要求.md](docs/资政大模型-项目要求.md) | 结题目标、现状盘点（已有/缺失资产）、功能与非功能需求、验收标准 |
| [docs/资政大模型-结题任务清单.md](docs/资政大模型-结题任务清单.md) | 重建导向的任务清单（阶段 0 资产保全 + 模块 A-E） |
| [docs/资政大模型-结题报告.md](docs/资政大模型-结题报告.md) | 结题报告（架构/实施/成果/质量评估/目标对照） |
| [docs/项目开发Harness-流程与注意事项.md](docs/项目开发Harness-流程与注意事项.md) | AI 辅助开发通用流程框架与踩坑清单（可复用至其他项目） |
| [docs/资政大模型-部署与安全文档.md](docs/资政大模型-部署与安全文档.md) | D5 部署方案与安全合规检查报告（敏感词过滤、审计、隐私） |
| [docs/资政大模型-检索评估报告.md](docs/资政大模型-检索评估报告.md) | C3 DC-RAG 检索质量评估报告 |
| [docs/资政大模型--大一年度项目立项报告 (1).pdf](docs/资政大模型--大一年度项目立项报告%20(1).pdf) | 立项报告（立项背景、技术路线、进度安排） |
| [dify/Rag.yml](dify/Rag.yml) | Dify 应用配置（moonshot-v1-8k + 政务公文写作提示词） |

---

## 开发规范

1. **版本管理**：所有产出随做随提交（`git add <file>` → `git commit` → `git push`），防止资产再次丢失。
2. **目录约定**：源码入 `src/`，脚本入 `scripts/`，数据入 `data/`，文档入 `docs/`，Dify 配置入 `dify/`。
3. **提交规范**：`type: 简述`（如 `feat:`、`fix:`、`docs:`、`refactor:`），一行标题 + 可选正文。
4. **敏感信息**：API Key、凭据一律不提交，使用环境变量（见 `.gitignore` 中的 `.env`）。
5. **数据集备份**：Dify 数据集（`dify/Rag.yml` 中的 dataset_id）需导出备份到 `dify/` 目录。

---

## 当前进度

- [x] 阶段 0：资产保全（git 仓库 + 远程备份 https://github.com/Jianzi270/zizhengLLM + 目录规范）
- [x] 模块 A：数据层（A1 盘点 ✅ / A2 元数据 ✅ / A3 清洗流水线 ✅ / A4 爬虫 ✅）
- [x] 模块 B：知识库构建（B1 嵌入模型 ✅ / B2 文档切分 ✅ / B3 摘要 ✅ / B4 GMM 聚类 ✅ / B5 树形知识库 ✅ / B6 数据集导出备份 ✅）
- [x] 模块 C：DC-RAG 检索与生成（C1 输入增强 ✅ / C2 DC-RAG 检索 ✅ / C3 质量评估 ✅ / C4 生成 ✅）
- [x] 模块 D：Agent 与应用（D1-D4 自研智能体 Web 应用 ✅ / D5 安全与合规 ✅）
- [~] 模块 E：结题验收（E1 系统集成与联调 ✅ / E2 结题材料 ✅（演示视频待录制） / E3 经费）

进度明细见 [docs/资政大模型-结题任务清单.md](docs/资政大模型-结题任务清单.md)。

---

## 环境配置

### 1. 环境要求

- Python ≥ 3.9（64 位）
- 磁盘空间：依赖与模型缓存约 2-3 GB（torch + BGE 嵌入模型）
- 网络：安装依赖需联网；首次运行嵌入模型会自动从 HuggingFace 下载（约 100 MB，缓存到 `models/embedding/`）

### 2. 安装依赖

```bash
# 可选：创建并激活虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate

# 安装依赖（网络受限时加国内镜像，如清华源）
pip install -r requirements.txt
# 备选：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 依赖清单：`requests`、`beautifulsoup4`、`pymupdf`、`numpy`、`sentence-transformers`、`torch`、`flask`。

### 3. 配置 LLM API Key（.env）

项目使用 OpenAI 兼容接口调用生成大模型（DeepSeek / Moonshot / OpenAI 等），**只需生成环节的 Key**，检索环节完全本地。

```bash
# 1. 复制模板为 .env（.env 已被 .gitignore 忽略，不会提交）
copy .env.example .env
# 2. 编辑 .env，填入三个变量：
#    LLM_API_KEY=sk-你的Key          # 必填
#    LLM_BASE_URL=https://api.deepseek.com/v1   # 服务地址（DeepSeek/Moonshot/OpenAI 等）
#    LLM_MODEL=deepseek-chat         # 模型名（如 deepseek-chat、moonshot-v1-8k、gpt-4o-mini）
```

> 本项目实测使用：`LLM_BASE_URL=https://api.deepseek.com/v1`、`LLM_MODEL=deepseek-v4-flash`（推理模型，`max_tokens` 需 ≥1500，已在代码中配置）。
>
> **注意**：`.env` 含密钥，绝不提交；`LLM_API_KEY` 缺失时，检索/清洗等本地流程仍可用，仅增强与生成环节会回退或报错提示。

### 4. 验证环境就绪

```bash
# 嵌入模型自测：加载 BGE 模型并验证相似度（首次运行会下载模型）
python -m src.embed.embed --test

# 端到端联调（快速模式，不调 LLM）：数据层/知识库/检索/应用逐层检查
python scripts/e2e_check.py
```

自检输出示例：`向量形状: (3, 512)`、`相似度自检通过 ✓`，即环境就绪。

---

## 运行智能体应用

> **说明**：本项目以自研 DC-RAG 链路替代 Dify 平台的 API Key 依赖——检索环节完全本地（BGE 嵌入 + 层次化知识库），仅生成环节调用 `.env` 中的 LLM Key（DeepSeek）。

```bash
# 1. 环境配置见上方"环境配置"章节（依赖安装 + .env 填写）
# 2. 启动服务
python app/server.py
# 3. 浏览器访问 http://127.0.0.1:8000 开始对话
```

数据集导出备份（B6 交付物）：

```bash
python scripts/export_dify_dataset.py                 # 导出 Dify 兼容数据集（data/export/dify_dataset.jsonl）
python scripts/export_dify_dataset.py --format chunks # 导出文本块备份（data/export/chunks_backup.jsonl）
```

一键爬取 + 清洗 + 入知识库（手动执行，不自动调度、不执行 git 操作）：

```bash
python scripts/auto_update.py                # 一键：爬取 → 清洗 → 摘要 → 切分 → 聚类 → 重建知识库 → 验证检索
python scripts/auto_update.py --dry-run      # 仅预览候选，不下载不清洗
python scripts/auto_update.py --skip-crawl   # 跳过爬取，仅清洗+更新知识库
python scripts/auto_update.py --skip-kb      # 仅爬取+清洗，不更新知识库
# 也可作为代码接口调用：from scripts.auto_update import run_pipeline; run_pipeline()
```

> 流水线将新数据完整送入知识库（新增文档生成摘要、重建三级索引）并可被 DC-RAG 检索到；残缺文件（无正文）自动跳过。
> 运行后输出新增/残缺提醒，由人工审核后再手动 `git add/commit/push`，脚本不会自动推送。

> 若日后有 Dify 平台环境，可将 `data/export/dify_dataset.jsonl` 导入 Dify 数据集，并使用 `dify/Rag.yml` 作为平台侧应用（备选方案）。
