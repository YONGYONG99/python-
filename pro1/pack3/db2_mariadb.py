# 원격 데이터베이스 연동 프로그램
# pip install mysqlclient

import MySQLdb

# 연결 잘 됐는지 확인
# conn = MySQLdb.connect(host = '127.0.0.1', user = 'root', password='123', database='test')
# print(conn)
# conn.close()

#sangdata table과 연동
config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8',
    'use_unicode':True
}

try:
    conn = MySQLdb.connect(**config) # ** -> dict 받을때 사용
    # print(conn) # 연동 잘 됐는지 확인
    cursor = conn.cursor() # 연결객체 만들기
    
    # insert
    
    # sql = "insert into sangdata(code,sang,su,dan) values(10,'신상1',5,'5000')"
    # cursor.execute(sql) # 여기까지 해도 안들어감 , 자동 commit이 안되기 떄문
    
    # 이 방법도 있음
    # sql = "insert into sangdata values(%s %s %s %s)" 
    # sql_data = '11','아아' , 12 ,5500
    # count = cursor.execute(sql, sql_data)
    # print(count)
    
    # conn.commit() # commit 해주기 
    # 두번 하면 err / err : (1062, "Duplicate entry '10' for key 'PRIMARY'")
    
    # update
    sql = "update sangdata set sang=%s,su=%s where code=%s"
    sql_data = ('파이썬' ,50 ,10)
    count = cursor.execute(sql, sql_data)
    print(count)
    conn.commit()
    
    # delete
    # code = '4'
    # sql = "delete from sangdata where code=" + code  # secure coding 가이드에 위배
    # 아래처럼 하자
    # sql = "delete from sangdata where code='{0}'".format(code)
    # cursor.execute(sql)
    # 또는 이렇게
    # sql = "delete from sangdata where code=%s"
    # cursor.execute(sql, (code,))
    # conn.commit()
    
    # select
    sql = "select code, sang , su , dan  from sangdata"
    cursor.execute(sql) 
    
    
    # 방법1
    for data in cursor.fetchall():
        # print(data)
        print('%s %s %s %s '%data)
        
    # 방법2
    print()
    for r in cursor:
        # print(r)
        print(r[0],r[1],r[2],r[3])
    
    # 방법3
    print()
    for (code,sang,su,dan) in cursor:
        print(code,sang,su,dan)
    
    # 방법3-1
    print()
    for (a,품명,su,kbs) in cursor:
        print(a,품명,su,kbs)
        
except Exception as e:
    print('err :',e)
finally:
    cursor.close()
    conn.close()


