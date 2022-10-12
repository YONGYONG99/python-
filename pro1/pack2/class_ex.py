# 클래스의 포함관계 연습문제 

class CoinIn:
    def calc(self, cupCount):
        re = ""
        
        if self.coin < 200:
            re = "요금이 부족하네요"
        elif cupCount * 200 > self.coin:
            re = "요금이 부족하네요"
        else:
            self.change = self.coin - (200 * cupCount)  # 잔돈 계산
            re = "커피 {}잔과 잔돈 {}원".format(cupCount, self.change)

        return re

class Machine():
    cupCount = 1  # 현재 코드에서는 의미 없음

    def __init__(self):
        self.coinIn = CoinIn()  # 포함

    def showData(self):
        self.coinIn.coin = int(input("동전을 입력하세요 :"))
        self.cupCount = int(input("몇 잔을 원하세요 :"))

        print(self.coinIn.calc(self.cupCount))


if __name__ == '__main__':
    Machine().showData()