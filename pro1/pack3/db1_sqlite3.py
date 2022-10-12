# 개인용 DB : sqlite3 : 파이썬 기본 개인용 데이터베이스

import sqlite3
print(sqlite3.sqlite_version)

print()
# conn = sqlite3.connect('exam.db')
conn = sqlite3.connect(':memory:') # ram에 일시적으로 data가 저장됨 - 테스트용

try:
    cursor = conn.cursor() # SQL문 처리
    
    # 테이블 생성
    cursor.execute("create table if not exists fritab(name text, phone text)")
    
    # 자료 추가
    cursor.execute("insert into fritab(name,phone) values('한국인','111-1111')")
    cursor.execute("insert into fritab(name,phone) values('우주인','222-1111')")
    cursor.execute("insert into fritab(name,phone) values(?,?)" , ('신기해' , '333-1111'))
    inputdata = ('신기루' , '444-1111')
    cursor.execute("insert into fritab values(?,?)", inputdata)
    conn.commit()
    
    # select 
    cursor.execute("select * from fritab")
    print(cursor.fetchone())
    print(cursor.fetchall())
    
    
    
    
except Exception as e:
    print('err : ', e)
    conn.rollback()
finally:
    conn.close()