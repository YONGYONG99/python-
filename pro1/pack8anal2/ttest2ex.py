# [two-sample t 검정 : 문제1] 
# 다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
# 포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.
#    blue : 70 68 82 78 72 68 67 68 88 60 80
#    red : 60 65 55 58 67 59 61 68 77 66 66

import pandas as pd
from scipy import stats
from numpy import average
import numpy as np
from dask.array.random import random
import pickle
from partd.pickle import Pickle
import numpy as np
import scipy.stats as stats


blue = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
red = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]

# 귀무 : 포장지 색상에 따른 제품의 매출액에 차이가 없다.
# 대립 : 포장지 색상에 따른 제품의 매출액에 차이가 있다.
print(average(blue), ' ', average(red))
two_sample = stats.ttest_ind(blue, red)
print(two_sample)
# statistic=2.9280203225212174, pvalue=0.008316545714784403
# 해석 : pvalue=0.008316 < 0.05 이므로 귀무가설 기각.
# 포장지 색상에 따른 제품의 매출액에 차이가 있다.

print('-----------------------------------------------')

# [two-sample t 검정 : 문제2]  
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
#   남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
#   여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4

# 귀무 : 남녀간 혈관 내의 콜레스테롤 양에 차이가 없다
# 대립 : 남녀간 혈관 내의 콜레스테롤 양에 차이가 있다
np.random.seed(123)
male = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
female = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4] 

male_ran = np.random.choice(male, size=15, replace=False)
female_ran = np.random.choice(female, size=15, replace=False)
print(male_ran,female_ran)
print(average(male_ran), ' ', average(female_ran))

# 정규성 확인
print(stats.shapiro(male_ran).pvalue) # 0.10370098054409027
print(stats.shapiro(female_ran).pvalue) # 0.0005095459055155516

two_sample2 = stats.ttest_ind(male_ran, female_ran)
print(two_sample2)
# statistic=0.5541664126702524, pvalue=0.5838642862135702
# 해석 : pvalue=0.5838642 > 0.05 이므로 귀무가설 채택.
# 남녀간 혈관 내의 콜레스테롤 양에 차이가 없다.
print(stats.wilcoxon(male_ran, female_ran)) # 정규성 만족 않하므로 wilcoxon 사용해야함


print('-------------------------------------------')

# [two-sample t 검정 : 문제3]
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오. 총무 10 , 영업 20
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.

# 귀무 :  총무부, 영업부 직원의 연봉의 평균에 차이가 존재하지 않는다.
# 대립 :  총무부, 영업부 직원의 연봉의 평균에 차이가 존재한다.

import MySQLdb
# 위에 pack3에 mydb.dat 복붙해오고 , db3.mariadb.py 에 코드문 가져오자
with open('mydb.dat' , mode='rb') as obj:
    config = pickle.load(obj)

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = "select buser_num, jikwon_pay from jikwon where buser_num = 10 or buser_num = 20"
    cursor.execute(sql)
    print(cursor)
    
    ar = []
    for data in cursor.fetchall():
        ar.append((data[0],data[1]))
    # print(ar)
    df = pd.DataFrame(ar, columns = ['부서','연봉'])
    print(df.head(10))
    pay1 = np.array(df[df['부서'] == 10])[:, 1]
    pay2 = np.array(df[df['부서'] == 20].연봉)
    print(pay1)
    print(pay2)
    print(average(pay1), ' ', average(pay2))
    two_sample3 = stats.ttest_ind(pay1, pay2) # 두 개의 표본에 대한 t-test 실시
    print(two_sample3)
    
except Exception as e:
    print('err : ',e)
finally:
    cursor.close()
    conn.close()

# statistic=0.4585177708256519, pvalue=0.6523879191675446
# 해석 : pvalue=0.65238 > 0.05 이므로 귀무가설 채택.
# 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하지 않는다.

print('-----------------------------------------------------------')

# [대응표본 t 검정 : 문제4]
# 어느 학급의 교사는 매년 학기 내 치뤄지는 시험성적의 결과가 실력의 차이없이 비슷하게 유지되고 있다고 말하고 있다. 
# 이 때, 올해의 해당 학급의 중간고사 성적과 기말고사 성적은 다음과 같다. 점수는 학생 번호 순으로 배열되어 있다.
#    중간 : 80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80
#    기말 : 90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95
# 그렇다면 이 학급의 학업능력이 변화했다고 이야기 할 수 있는가?

# 귀무 : 이 학급의 학업 능력은 변화하지 않았다.
# 대립 : 이 학급의 학업 능력은 변화했다.

middle = [80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80]
final = [90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95]

print(np.mean(middle), ' ', np.mean(final)) # 74.1666 vs  81.6666
print(stats.ttest_rel(middle, final))
# 해석 : pvalue=0.0234 < 0.05 이므로 귀무 채택
# 이 학급의 학업 능력은 변화했따