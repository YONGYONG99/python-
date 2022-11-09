# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.

# kind quantity
# 1 64
# 2 72
# 3 68
# 4 77
# 2 56
# 1 NaN
# 3 95
# 4 78
# 2 55
# 1 91
# 2 63
# 3 49
# 4 70
# 1 80
# 2 90
# 1 33
# 1 44
# 3 55
# 4 66
# 2 77
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

# 귀무 : 기름 종류에 따라 흡수하는 기름의 평균 차이가 존재하지 않는다.
# 대립 : 기름 종류에 따라 흡수하는 기름의 평균 차이가 존재한다.

data = {
    'kind':[1,2,3,4,2,1,3,4,2,1,2,3,4,1,2,1,1,3,4,2],
    'quantity':[64,72,68,77,56,np.nan,95,78,55,91,63,49,70,80,90,33,44,55,66,77]
}
frame = pd.DataFrame(data)
frame = frame.fillna(frame.mean())
print(frame)

kind1 = np.array(frame[frame['kind'] == 1])[:, 1]
kind2 = np.array(frame[frame['kind'] == 2])[:, 1]
kind3 = np.array(frame[frame['kind'] == 3])[:, 1]
kind4 = np.array(frame[frame['kind'] == 4])[:, 1]
print(kind2)
print(kind1)

# 정규성 확인
print(stats.shapiro(kind1).pvalue) # 0.868 > 0.05 정규성 만족
print(stats.shapiro(kind2).pvalue) 
print(stats.shapiro(kind3).pvalue)
print(stats.shapiro(kind4).pvalue)
    
# 등분산성 확인
print(stats.levene(kind1, kind2, kind3, kind4).pvalue) # 0.3268 > 0.05 등분산성 만족

# anova 
print()
print(stats.f_oneway(kind1,kind2,kind3,kind4))
# F_onewayResult(statistic=0.26693511759829797, pvalue=0.8482436666841788)
# pvalue=0.8482 > 0.05 이므로 귀무 채택
# 기름 종류에 따라 흡수하는 기름의 평균 차이가 존재하지 않는다.

# 사후 검증
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkeyResult = pairwise_tukeyhsd(endog=frame.quantity, groups=frame.kind)
print(turkeyResult)  # 모두 (reject False) 1,2,3,4 기름 종류간의 평균차이가 유의미하지 않다

# 시각화
turkeyResult.plot_simultaneous(xlabel='mean',ylabel='group')
plt.show() 


print('예제 2')
# [ANOVA 예제 2]
# DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오.
# 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

# 귀무 : 부서별 연봉 평균의 차이가 없다.
# 대립 : 부서별 연봉 평균의 차이가 있다.

import MySQLdb
import pickle
from partd.pickle import Pickle
from numpy import average

with open('mydb.dat' , mode='rb') as obj:
    config = pickle.load(obj)

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = "select buser_num, jikwon_pay from jikwon"
    cursor.execute(sql)
    print(cursor)
    
    ar = []
    for data in cursor.fetchall():
        ar.append((data[0],data[1]))
    print(ar)
    df = pd.DataFrame(ar, columns = ['부서','연봉'])
    print(df.head(10))
    pay1 = np.array(df[df['부서'] == 10])[:, 1]
    pay2 = np.array(df[df['부서'] == 20].연봉)
    pay3 = np.array(df[df['부서'] == 30].연봉)
    pay4 = np.array(df[df['부서'] == 40].연봉)
    print(pay1)
    print(pay2)
    print(pay3)
    print(pay4)
    print(average(pay1), ' ', average(pay2), ' ', average(pay3), ' ', average(pay4))
    
    # 정규성 확인
    print(stats.shapiro(pay1).pvalue) # 0.0260 < 0.05 정규성 만족하지 않는다.
    print(stats.shapiro(pay2).pvalue) # 0.0256
    print(stats.shapiro(pay3).pvalue) # 0.4194
    print(stats.shapiro(pay4).pvalue) # 0.9078
    
    # 등분산성 확인
    print(stats.levene(pay1, pay2, pay3, pay4).pvalue) # 0.79807 > 0.05 등분산성 만족
    
    # anova 
    print()
    print(stats.f_oneway(pay1,pay2,pay3,pay4))
    # F_onewayResult(statistic=0.41244077160708414, pvalue=0.7454421884076983)
    # pvalue=0.7454 > 0.05 이므로 귀무 채택
    # 부서별 연봉 평균의 차이가 없다.
    
    # 정규성 불만족 하므로 kruskal 사용
    print(stats.kruskal(pay1,pay2,pay3,pay4))
    # KruskalResult(statistic=1.671252253685445, pvalue=0.6433438752252654)
    # pvalue=0.64334 > 0.05 이므로 여전히 귀무 채택
    
    # 사후 검증
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    turkeyResult = pairwise_tukeyhsd(endog=df.연봉, groups=df.부서)
    print(turkeyResult)  # 모두 (reject False) 1,2,3,4 부서간의 연봉평균의 차이는 유의미하지 않다.
    
    # 시각화
    turkeyResult.plot_simultaneous(xlabel='mean',ylabel='group')
    plt.show() 
    
except Exception as e:
    print('err : ',e)
finally:
    cursor.close()
    conn.close()
