# file i/o + with 문

try:
    # 저장
    with open('file_test3.txt', mode='w', encoding='utf8') as obj1:
        obj1.write('파이썬으로 문서 저장\n')
        obj1.write('with문을 쓰면\n')
        obj1.write('명시적으로 close()를 하지 않는다.')
        
    # 읽기
    with open('file_test3.txt', mode='r', encoding='utf8') as obj2:
        print(obj2.read())

except Exception as e:
    print('오류 : ' , e)

print('---피클링(객체 저장)---')
import pickle
    
try:
    dictData = {'소현':'111-1111','승경':'222-2222'}
    listData = ['곡물그대로리','새우깡']
    tupleData = (listData , dictData) # 복합 객체
    
    # 개체를 저장
    with open('hello.dat' , mode = 'wb') as obj3:
        pickle.dump(tupleData, obj3)
        pickle.dump(listData, obj3)
    
    # 개체를 읽기
    with open('hello.dat' , mode = 'rb') as obj3:
        a , b = pickle.load(obj3)
        print(a)
        print(b)
        c = pickle.load(obj3)
        print(c)
    
except Exception as e2:
    print('오류2 : ', e2)