"""资政大模型 — 自研 DC-RAG 智能体 Web 服务。

架构（全部本地/自研，仅生成环节使用 .env 中的 LLM Key）：
  用户问题 → 安全合规检查（敏感词过滤）→ C1 LLM 输入增强 → C2 DC-RAG 三级检索
          → C4 LLM 生成结构化答案 → 输出合规检测 → 展示（含来源文档）→ 审计日志

启动：
  python app/server.py            # 访问 http://127.0.0.1:8000
"""
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from src.generate.answer import TASK_MODES, generate_answer  # noqa: E402
from src.security.compliance import audit_log, check_content  # noqa: E402

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
MAX_QUESTION_CHARS = 1000


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    """问答接口：question -> {answer, sources, enhanced}（含输入/输出合规检测与审计）"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "请求体必须是 JSON 对象"}), 400
    raw_question = data.get("question")
    if raw_question is not None and not isinstance(raw_question, str):
        return jsonify({"error": "question 必须是字符串"}), 400
    question = (raw_question or "").strip()
    if not question:
        return jsonify({"error": "请输入问题"}), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({"error": f"问题长度不能超过 {MAX_QUESTION_CHARS} 个字符"}), 400
    mode = data.get("mode", "qa")
    if mode not in TASK_MODES:
        return jsonify({"error": "不支持的任务模式"}), 400

    # 输入侧合规：命中敏感词直接拒绝（输出可控）
    in_hits = check_content(question)
    if in_hits:
        audit_log({"event": "blocked", "reason": "input_sensitive", "question": question, "hits": in_hits})
        return jsonify({"error": "输入包含不合规内容，已拒绝处理。"}), 400

    t0 = time.time()
    try:
        r = generate_answer(question, top_c=2, top_d=3, top_k=3, mode=mode)
        # 输出侧合规：命中敏感词则标记，不向用户返回（可审计）
        out_hits = check_content(r["answer"])
        if out_hits:
            audit_log({"event": "blocked", "reason": "output_sensitive", "question": question,
                       "hits": out_hits, "answer_excerpt": r["answer"][:200]})
            return jsonify({"error": "生成内容未通过合规检测，已拦截。请调整提问方式。"}), 500
        audit_log({"event": "ask", "question": question, "enhanced": r["enhanced"],
                   "mode": mode,
                   "sources": [s["doc_id"] for s in r["sources"]],
                   "cost_s": round(time.time() - t0, 2), "answer_chars": len(r["answer"])})
        return jsonify({
            "answer": r["answer"],
            "sources": r["sources"],
            "enhanced": r["enhanced"],
            "mode": mode,
        })
    except Exception as e:  # 网络/Key 错误友好提示
        error_id = uuid.uuid4().hex[:12]
        audit_log({"event": "error", "error_id": error_id, "question": question, "error": str(e)[:500],
                    "cost_s": round(time.time() - t0, 2)})
        return jsonify({"error": f"生成服务暂时不可用，请稍后重试。错误编号：{error_id}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    try:
        from src.retrieval.kb import load_kb
        kb = load_kb()
        return jsonify({"status": "ready", "documents": len(kb["index"]["doc_ids"]),
                        "chunks": len(kb["index"]["chunks"])})
    except Exception:
        return jsonify({"status": "not_ready"}), 503


if __name__ == "__main__":
    print("资政大模型智能体已启动：http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)
