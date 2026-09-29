# -*- coding: utf-8 -*-
"""RAG 核心链路完整测试：上传 → 问答 → Token记录 → 导出"""
import json
import httpx

BASE = "http://127.0.0.1:8000"


def main():
    client = httpx.Client(timeout=120)

    # 登录
    r = client.post(BASE + "/auth/login", json={"username": "test_user", "password": "test123456"})
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("1 LOGIN:", r.status_code)

    # 上传测试文档
    test_doc = """RAG智能文档问答系统技术说明
检索增强生成（RAG）是一种结合信息检索与大语言模型的问答技术。
系统流程：文档上传 → 文本切分 → 向量化 → 混合检索 → 重排序 → 生成回答。
混合检索包含向量语义检索（BGE嵌入模型）与关键词检索（SQLite FTS5全文索引）。
FTS5使用trigram分词器，适合中文关键词检索。检索结果经BGE Reranker精排后送入大模型生成。
JWT认证模块基于python-jose与bcrypt实现，支持Token自动刷新。
报告导出支持WeasyPrint生成PDF文档与python-docx生成Word文档。
"""
    files = {"file": ("测试文档.txt", test_doc.encode("utf-8"), "text/plain")}
    r = client.post(BASE + "/upload", headers=headers, files=files)
    print("2 UPLOAD:", r.status_code, r.json())

    # 问答（非流式）
    r = client.post(BASE + "/chat", headers=headers, json={"query": "这个系统用什么技术做关键词检索？", "use_reranker": True})
    print("3 CHAT:", r.status_code)
    if r.status_code == 200:
        data = r.json()
        print("   answer:", data["answer"][:150].replace("\n", " "))
        print("   provider:", data.get("provider"), "| token_usage:", data.get("token_usage"))
        print("   sources:", len(data.get("sources", [])))
    else:
        print("   error:", r.text[:300])

    # Token统计（应新增一条）
    r = client.get(BASE + "/stats/token-usage?days=7", headers=headers)
    total = sum(p["total_tokens"] for p in r.json()["data"])
    calls = sum(p["calls"] for p in r.json()["data"])
    print("4 TOKEN-STATS: 总Token=", total, "调用次数=", calls)

    # 导出 DOCX
    r = client.post(
        BASE + "/export/docx",
        headers=headers,
        json={
            "question": "这个系统用什么技术做关键词检索？",
            "answer": "使用SQLite FTS5全文索引，trigram分词器。",
            "sources": [{"filename": "测试文档.txt", "page": 1, "score": 0.95, "content_preview": "FTS5使用trigram分词器"}],
            "title": "RAG 测试报告",
        },
    )
    print("5 EXPORT DOCX:", r.status_code, "size=", len(r.content) if r.status_code == 200 else r.text[:200])

    # 导出 PDF（Windows 无 GTK 时预期友好报错）
    r = client.post(
        BASE + "/export/pdf",
        headers=headers,
        json={"question": "q", "answer": "a", "sources": []},
    )
    print("6 EXPORT PDF:", r.status_code)
    if r.status_code != 200:
        print("   detail:", r.json().get("detail", "")[:200])

    # 流式问答（SSE）
    with client.stream("POST", BASE + "/chat/stream", headers=headers, json={"query": "你好", "use_reranker": True}) as resp:
        print("7 STREAM status:", resp.status_code)
        chunks = 0
        for line in resp.iter_lines():
            if line.startswith("data: "):
                chunks += 1
        print("   SSE events:", chunks)

    client.close()


if __name__ == "__main__":
    main()
