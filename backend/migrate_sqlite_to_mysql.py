# -*- coding: utf-8 -*-
"""
migrate_sqlite_to_mysql.py - SQLite → MySQL 数据迁移（一次性）
============================================================
将 SQLite 旧库（backend/data/rag.db）中的业务数据迁移到 MySQL（rag_kb）。
表：users / documents / chat_sessions / chat_messages / token_usage
使用方式（在 backend/ 目录下）:
    python migrate_sqlite_to_mysql.py
说明：目标表若已有相同主键则跳过（ON DUPLICATE KEY UPDATE id=id），可安全重复运行。
"""
import json
import sqlite3
import sys
from pathlib import Path

import pymysql

BASE_DIR = Path(__file__).parent.resolve()
SQLITE_PATH = BASE_DIR / "data" / "rag.db"

# 从 backend/config.yaml 读取 MySQL 连接信息（避免在仓库中硬编码密码）
# config.yaml 属于本地敏感配置（已被 .gitignore 排除），不随仓库上传
try:
    from config import settings
    import re

    _url = settings.database.url  # mysql+pymysql://user:pass@host:port/db?charset=...
    _m = re.match(r"mysql\+pymysql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)", _url)
    if _m:
        MYSQL = dict(
            host=_m.group(3),
            port=int(_m.group(4)),
            user=_m.group(1),
            password=_m.group(2),
            database=_m.group(5),
            charset="utf8mb4",
        )
    else:
        raise ValueError(f"无法解析 database.url: {_url}")
except Exception as _e:  # 读不到配置时给出明确提示，不静默降级
    print(f"[错误] 无法从 config.yaml 读取 MySQL 配置: {_e}")
    print("请先在 backend/config.yaml 中配置 database.url（mysql+pymysql://用户:密码@主机:端口/rag_kb）")
    sys.exit(1)

# 迁移顺序：外键依赖的父表在前
TABLES = [
    "users",          # 无外键
    "documents",      # FK -> users.id
    "chat_sessions",  # FK -> users.id
    "chat_messages",  # 业务键 session_id（无FK约束）
    "token_usage",    # FK -> users.id
]


def rows_from_sqlite(conn: sqlite3.Connection, table: str):
    """读取 SQLite 表的全部行（保持主键顺序）"""
    cur = conn.execute(f'SELECT * FROM "{table}" ORDER BY 1')
    cols = [d[0] for d in cur.description]
    for row in cur.fetchall():
        yield dict(zip(cols, row))


def to_mysql_value(table: str, col: str, value):
    """按列类型做值转换（JSON / 空串处理）"""
    if value is None:
        return None
    # chat_messages.sources 为 JSON 列：SQLite 中存 JSON 文本，需校验
    if table == "chat_messages" and col == "sources":
        if isinstance(value, str):
            s = value.strip()
            if not s:                     # 空串 -> NULL
                return None
            try:
                json.loads(s)             # 校验合法性
                return s
            except json.JSONDecodeError:
                return None               # 非法JSON -> NULL
        return json.dumps(value, ensure_ascii=False)
    return value


def main():
    if not SQLITE_PATH.exists():
        print(f"❌ 未找到 SQLite 库: {SQLITE_PATH}")
        sys.exit(1)

    # ---------- 连接两库 ----------
    sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
    sqlite_conn.text_factory = str  # 防止中文/二进制编码异常
    mysql_conn = pymysql.connect(**MYSQL, autocommit=False)
    mysql_cur = mysql_conn.cursor()

    print("=" * 60)
    print(f"SQLite 旧库: {SQLITE_PATH}")
    print(f"MySQL 目标: rag_kb@{MYSQL['host']}:{MYSQL['port']}")
    print("=" * 60)

    total = {}
    try:
        for table in TABLES:
            # 取 MySQL 目标表的列（用于过滤 SQLite 中多出的列）
            mysql_cur.execute(
                f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s", (MYSQL["database"], table)
            )
            mysql_cols = [r[0] for r in mysql_cur.fetchall()]

            rows = list(rows_from_sqlite(sqlite_conn, table))
            insert_cols = [c for c in mysql_cols]
            col_placeholders = ", ".join(["%s"] * len(insert_cols))
            insert_cols_sql = ", ".join(insert_cols)

            migrated = 0
            for r in rows:
                values = [
                    to_mysql_value(table, c, r.get(c))
                    for c in insert_cols
                ]
                try:
                    mysql_cur.execute(
                        f"INSERT INTO `{table}` ({insert_cols_sql}) VALUES ({col_placeholders}) "
                        f"ON DUPLICATE KEY UPDATE `id`=`id`",   # 主键冲突时跳过
                        values,
                    )
                    migrated += 1
                except pymysql.MySQLError as e:
                    print(f"  ⚠️  {table} 行 id={r.get('id')} 失败: {e}")
            total[table] = migrated
            print(f"  ✓ {table:<16} SQLite={len(rows):<6} MySQL写入={migrated}")
            mysql_conn.commit()

        print("-" * 60)
        print("✅ 迁移完成！各表合计写入:", sum(total.values()), "行")
        print("   提示：后端服务已运行在 MySQL 模式，请刷新前端重新登录。")
    finally:
        sqlite_conn.close()
        mysql_cur.close()
        mysql_conn.close()


if __name__ == "__main__":
    main()
