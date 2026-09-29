# -*- coding: utf-8 -*-
"""RAG API 端到端冒烟测试脚本"""
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"


def post(path, body, token=None, timeout=25):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
    except Exception as e:
        return -1, repr(e)


def get(path, token=None, timeout=25):
    req = urllib.request.Request(BASE + path, headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
    except Exception as e:
        return -1, repr(e)


if __name__ == "__main__":
    # 1. 注册
    s, body = post("/auth/register", {"username": "test_user", "email": "test@example.com", "password": "test123456"})
    print("1 REGISTER:", s, list(body.keys()) if isinstance(body, dict) else body)

    # 2. 登录
    s, login = post("/auth/login", {"username": "test_user", "password": "test123456"})
    token = login.get("access_token", "")
    refresh = login.get("refresh_token", "")
    print("2 LOGIN:", s, "access_len=", len(token), "refresh_len=", len(refresh))

    # 3. 当前用户
    s, me = get("/auth/me", token)
    print("3 ME:", s, me if isinstance(me, dict) else me)

    # 4. 无Token访问 /docs（应401）
    try:
        req = urllib.request.Request(BASE + "/docs")
        with urllib.request.urlopen(req, timeout=15) as r:
            print("4 NO-TOKEN /docs ->", r.status, "(期望401，但实际200)")
    except urllib.error.HTTPError as e:
        print("4 NO-TOKEN /docs ->", e.code, "body=", repr(e.read().decode()[:200]))
    except Exception as e:
        print("4 NO-TOKEN /docs -> OTHER:", repr(e))

    # 5. 带Token访问 /docs
    try:
        req = urllib.request.Request(BASE + "/docs", headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode()
            docs = json.loads(raw)
            print("5 DOCS:", r.status, "文档数=", len(docs) if isinstance(docs, list) else docs)
    except urllib.error.HTTPError as e:
        print("5 DOCS ->", e.code, "body=", repr(e.read().decode()[:300]))
    except Exception as e:
        print("5 DOCS -> OTHER:", repr(e))

    # 6. 刷新Token
    s, ref = post("/auth/refresh", {"refresh_token": refresh})
    print("6 REFRESH:", s, "new_access_len=", len(ref.get("access_token", "")) if isinstance(ref, dict) else ref)

    # 7. Token用量统计
    s, stats = get("/stats/token-usage?days=7", token)
    print("7 TOKEN-STATS:", s, "点数=", len(stats.get("data", [])) if isinstance(stats, dict) else stats)

    # 8. 会话列表
    s, sessions = get("/stats/sessions", token)
    print("8 SESSIONS:", s, "会话数=", len(sessions) if isinstance(sessions, list) else sessions)
