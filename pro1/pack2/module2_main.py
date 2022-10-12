# 사용자 작성 모듈
print('뭔가를 하다가...')

# 다른 모듈의 멤버 호출

list1 = [1,3]
list2 = [2,4]

import pack2.mymodule1
pack2.mymodule1.listHap(list1, list2)
print(pack2.mymodule1.__file__)
print(pack2.mymodule1.__name__)

def abcd():
    if __name__ == '__main__':
        print('난 최상위 메인 모듈이야')
    
abcd()

print('가격은 {}원'.format(pack2.mymodule1.price))

#번거로우니까 좀 편한방법 (자주 가져온다면 이방법을)
from pack2.mymodule1 import price
print('가격은 {}원'.format(price))

from pack2.mymodule1 import kbs, mbc
kbs()
mbc()

print('\n다른 패키지에 있는 모듈 읽기')
import etc.mymodule2
print(etc.mymodule2.Hap(5, 3))

from etc.mymodule2 import Cha
print(Cha(5, 3))

print('\n다른 패키지(path가 설정된)에 있는 모듈 읽기')
# C:\work\anaconda3\Lib\site-packages
import mymodule3 # 패키지를 안써줘도 됨
print(mymodule3.Gop(5, 3))
from mymodule3 import Nanugi
print(Nanugi(5, 3))