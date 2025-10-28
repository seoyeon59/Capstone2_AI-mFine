import sqlite3

conn = sqlite3.connect("capstone2.db")
c = conn.cursor()

# 테이블 목록
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(c.fetchall())

# 테이블 구조 확인
print("\n=== user table ===")
c.execute("PRAGMA table_info(users);")
print(c.fetchall())

print("\n=== cameras table ===")
c.execute("PRAGMA table_info(cameras);")
print(c.fetchall())

print("\n=== ai_learning table ===")
c.execute("PRAGMA table_info(ai_learning);")
print(c.fetchall())

print("\n=== realtime_screen table ===")
c.execute("PRAGMA table_info(realtime_screen);")
print(c.fetchall())

conn.close()
