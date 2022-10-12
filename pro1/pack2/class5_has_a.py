# 자원의 재활용 : 클래스는 다른 클래스를 불러다 사용 가능
# 클래스의 포함관계(has a)

class Pohamhandle: # 핸들이 필요한 어떤 클래스에서든 호출될 수 있다.
    quantity = 0 # 회전량

    def LeftTurn(self, quantity):
        self.quantity = quantity
        return '좌회전'

    def RightTurn(self, quantity):
        self.quantity = quantity
        return '우회전'

# 자동차를 위한 여러 부품을 별도의 클래스로 제작 : 생략

# 완성차 클래스
class PohamCar:
    turnShowMessage = '정지'

    def __init__(self, ownerName):
        self.ownerName = ownerName
        self.handle = Pohamhandle() # 클래스의 포함

    def TurnHandle(self, q):
        if q > 0:
            self.turnShowMessage = self.handle.RightTurn(q)
        elif q < 0:
            self.turnShowMessage = self.handle.LeftTurn(q)
        elif q == 0:
            self.turnShowMessage = "직진"
            self.handle.quantity = 0

if __name__ == '__main__':
    tom = PohamCar('톰')
    tom.TurnHandle(10)
    print(tom.ownerName + '의 회전량은 ' + tom.turnShowMessage + str(tom.handle.quantity))
    tom.TurnHandle(0)
    print(tom.ownerName + '의 회전량은 ' + tom.turnShowMessage + str(tom.handle.quantity))
    print()
    oscar = PohamCar('오스카')
    oscar.TurnHandle(-5)
    print(oscar.ownerName + '의 회전량은 ' + oscar.turnShowMessage + str(oscar.handle.quantity))
