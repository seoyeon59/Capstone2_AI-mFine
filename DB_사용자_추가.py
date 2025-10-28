import sqlite3

conn = sqlite3.connect('capstone2.db')
c = conn.cursor()

# 기존 사용자 확인 후 없으면 추가
c.execute("SELECT id FROM users WHERE id = ?", ("test",))
if not c.fetchone():
    c.execute("""
    INSERT INTO users (id, password, username, observer_name)
    VALUES (?, ?, ?, ?)
    """, ("test", "0000", "전서연", "홍길동"))

# 기존 카메라 확인 후 없으면 추가
c.execute("SELECT user_id FROM cameras WHERE user_id = ?", ("test",))
if not c.fetchone():
    c.execute("""
    INSERT INTO cameras (user_id, camera_url)
    VALUES (?, ?)
    """, ("test", "http://192.168.45.3:8080/video"))

conn.commit()
conn.close()
