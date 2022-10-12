print('뭔가를 하다가 모듈의 멤버인 클래스를 선언하기')

class TestClass:    # new 없이 객체로 만들어줌 , prototype , 원형클래스 객체 생성 , 고유의 이름 공간을 확보
    aa = 1 # 멤버변수(멤버필드) , public

    def __init__(self): # 생성자
        print('생성자')
    
    def __del__(self):
        print('소멸자')

    def printMessage(self): # method
        name = '한국인' # 지역변수
        print(name)
        print(self.aa) # this 대신 self

print(TestClass, id(TestClass))
print(TestClass.aa)