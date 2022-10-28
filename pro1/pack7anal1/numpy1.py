# numpy :
# Numpy는 C언어로 구현된 파이썬 라이브러리로써, 고성능의 수치계산과 선형대수학을 위해 제작되었다.
# Numerical Python의 줄임말이기도 한 Numpy는 벡터 및 행렬 연산에 있어서 매우 편리한 기능을 제공.


# 변량 -> 표본데이터의 패턴을 파악할수있도록 통계량구할 수있다.??
# 모든 데이터를 대표할수있는 평균 , 평균으로 부터 얼마나 떨어져있는지 즉 분포 즉 분산(표준편차) 구할수있다.
# 파이썬은 자유도(n-1) 안쓰고 전체갯수를 사용함(R과 다른점) , 값도 다르게나옴 R이랑

grades = [1, 3, -2, 4] # 변량

# 아직 numpy사용하는거 아님
def grades_sum(grades):
    tot = 0
    for g in grades:
        tot += g
    return tot

def grades_avg(grades):
    tot = grades_sum(grades)
    ave = tot / len(grades)
    return ave

def grades_variance(grades): # 편차제곱의 평균 : 분산
    ave = grades_avg(grades)
    vari = 0
    for su in grades:
        vari += (su-ave) **2
    return vari / len(grades)
    # return vari / (len(grades) - 1) -> R에서 구하는 방식

def grades_std(grades):
    return grades_variance(grades) ** 0.5

print('합은 ' , grades_sum(grades))
print('평균은 ', grades_avg(grades))
print('분산은 ', grades_variance(grades))
print('표준편차 ', grades_std(grades))

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
#numpy 함수 사용해보자
import numpy
print('합은 ' , numpy.sum(grades))
print('평균은 ', numpy.mean(grades))
print('분산은 ', numpy.var(grades))
print('표준편차 ', numpy.std(grades))








