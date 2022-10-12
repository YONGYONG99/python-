# 키보드에서 부서번호를 입력받아 해당 부서 직원자료(사번, 이름, 부서, 연봉, 직급) 출력
import MySQLdb
# 아까 저장한 pickle 불러오기
import pickle

with open('mydb.dat' , mode='rb') as obj:
    config = pickle.load(obj)

def chulbal():
    try:
        conn = MySQLdb.connect(**config) # db 연결
        # print(conn)
        cursor = conn.cursor()
        buser_info = input('부서이름 : ')
        sql = """
            select jikwon_no, jikwon_name, buser_num, jikwon_pay, jikwon_jik
            from jikwon inner join buser 
            on jikwon.buser_num=buser.buser_no
            where buser_name='{0}'
        """.format(buser_info)
        # print(sql)
        
        cursor.execute(sql)
        datas = cursor.fetchall()
        # print(datas, len(datas))
        
        if len(datas) == 0:
            print(str(buser_info) + ' 에 해당되는 자료는 없어요')
            return # 이건 함수 탈출 # sys.exit(0) 이건 프로그램 강제종료
        
        for jikwon_no, jikwon_name,buser ,jikwon_pay,jikwon_jik in datas:
            print(jikwon_no, jikwon_name,buser,jikwon_pay,jikwon_jik)
        
        print('인원수 : {}'.format(len(datas)))
        
        
    except Exception as e:
        print('err : ', e)
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    chulbal()


