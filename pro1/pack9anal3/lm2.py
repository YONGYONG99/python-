# 방법4 : linregress를 사용. model O

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# iq에 따른 시험점수 예측
score_iq = pd.read_csv("../testdata/score_iq.csv")
print(score_iq.head())
#      sid  score   iq  academy  game  tv
# 0  10001     90  140        2     1   0
# 1  10002     75  125        1     3   3
# 2  10003     77  120        1     0   4
# 3  10004     83  135        2     3   2
# 4  10005     65  105        0     4   4
print(score_iq.info()) # score , iq 둘다 int64 
print(score_iq.corr()) # 0.882220 주목

x = score_iq.iq
y = score_iq.score

print(np.corrcoef(x, y)[0, 1]) # 0.8822203446134701 -> 피어슨 상관관계 / 양의 상관관계가 있군

# plt.scatter(x,y)
# plt.show() # 양의 상관관계가 보이는군

# 모델 생성
model = stats.linregress(x, y)
print(model)
# LinregressResult(slope=0.6514309527270075, intercept=-2.8564471221974657, rvalue=0.8822203446134699, pvalue=2.8476895206683644e-50, stderr=0.028577934409305443, intercept_stderr=3.546211918048538)
print('slope : ', model.slope) # slope :  0.6514309527270075
print('intercept : ', model.intercept) 
print('rvalue : ', model.rvalue) 
print('pvalue : ', model.pvalue) # 2.8476895206683644e-50 < 0.05 회귀모델은 유의하다. 두 변수 간에 인과관계가 있다.
print('stderr : ', model.stderr) 
# y_hat = 0.6514309527270075 * x + -2.8564471221974657

plt.scatter(x,y) # y -> 실제값 y
plt.plot(x, model.slope * x + model.intercept , c='red') #  model.slope*x + model.intercept -> 예측값 y
plt.show() # 회귀선 확인

# 점수 예측
print('점수 예측 : ' , model.slope * 140 + model.intercept) # 점수 예측 :  88.34388625958358
print('점수 예측 : ' , model.slope * 125 + model.intercept) # 점수 예측 :  78.57242196867847
print()
# linregress는 predict을 지원하지 않아서 numpy lib사용
new_df = pd.DataFrame({'iq':[140,125]})
print('점수 예측 : ' , np.polyval([model.slope, model.intercept], new_df))
# 점수 예측 :  [[88.34388626]
#            [78.57242197]]









































