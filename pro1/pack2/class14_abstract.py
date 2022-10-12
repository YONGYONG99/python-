from abc import *

class AbstractClass(metaclass = ABCMeta): # 추상 클래스가 됨
    @abstractmethod
    def myMethod(self): # 추상 메소드가 됨
        pass
    
    def normalMethod(self):
        print('추상클래스는 일반 메소드를 가질 수도 있다')

# parent = AbstractClass() # err / abstract class는 객체를 만들수 없음

class Child1(AbstractClass):
    name = '난 Child1'

    def myMethod(self):
        print('Child1에서 추상 메소드에 내용을 적음')

c1 = Child1() 
print(c1.name)
c1.myMethod()
c1.normalMethod()

print()

class Child2(AbstractClass):
    def myMethod(self):
        print('Child2에서 추상의 마법을 풀다')
        print('이제는 자유다')

    def normalMethod(self):
        print('추상 클래스의 일반 메소도는 오버라이딩이 선택적이다')
    
    def good(self):
        print('Child2 고유 메소드')

c2 = Child2()
c2.myMethod()
c2.normalMethod()
c2.good()

print('----------')
imsi = c1
imsi.myMethod()
print()
ismi = c2
imsi.myMethod()