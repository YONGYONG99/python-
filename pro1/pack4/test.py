# year = input('연도입력 : ')
#
# if int(year) % 4 == 0 and int(year) % 100 != 0 or int(year) % 400 == 0:
#     print(year + '년은 윤년')
# else:
#     print(year + '년은 평년')

# i = 0 
# while True:
#     if i<100:
#         i+=1
#
#     print(i, end=' ')
#     i += 1

# i = 1
# while i<=10:
#     j = 1
#     cnt = 1
#     re = '*' * 11
#     while j<=i:
#         # re = re - '*'
#         re = ' ' * (cnt-1) + '*' * (11 -cnt)
#         j+=1
#         cnt+=1
#     print(re)
#     i+=1

# print(' ' * 10 )

# i = 0
# while True:
#     if 1)__________:
#         i += 1
#         2)_________         
#
#     if i > 100: 3)________ 
#
#     print(i, end=' ')
#     i += 1

class Bicycle:
    def __init__(self,name,wheel,price):
        self.name = name
        self.wheel = wheel
        self.price = price
    
    def total(self):
        return self.wheel * self.price
    
    def display(self):
        print(str(self.name)+'님 자전거 바퀴 가격 총액은 ' + str(self.total()) + '원 입니다')

gildong = Bicycle('길동', 2, 50000)
gildong.display()

# import MySQLdb
#
# config = {
#     'host':'127.0.0.1',
#     'user':'root',
#     'password':'123',
#     'database':'test',
#     'port':3306,
#     'charset':'utf8',
#     'use_unicode':True
# }
#
#
# def chulbal():
#     try:
#         conn = MySQLdb.connect(**config) 
#         cursor = conn.cursor()
#         jik_info = input('직급 입력 : ')
#         sql = """
#             select jikwon_no, jikwon_name, jikwon_jik, buser_num
#             from jikwon
#             where jikwon_jik = '{0}'
#         """.format(jik_info)
#
#
#         cursor.execute(sql)
#         datas = cursor.fetchall()
#
#
#         if len(datas) == 0:
#             print(str(jik_info) + ' 에 해당되는 자료는 없어요')
#             return 
#
#         for jikwon_no, jikwon_name, jikwon_jik, buser_num in datas:
#             print(jikwon_no, jikwon_name, jikwon_jik, buser_num)
#
#
#     except Exception as e:
#         print('err : ', e)
#     finally:
#         cursor.close()
#         conn.close()
#
# if __name__ == '__main__':
#     chulbal()









