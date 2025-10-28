import sqlite3

conn = sqlite3.connect('capstone2.db')
c = conn.cursor()

tables = ['users', 'cameras', 'realtime_screen']

for table in tables:
    print(f"\n=== {table} 테이블 내용 ===")
    c.execute(f"SELECT * FROM {table}")
    rows = c.fetchall()
    if rows:
        for row in rows:
            print(row)
    else:
        print("데이터가 없습니다.")

conn.close()
