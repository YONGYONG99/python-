# file i/o

import os
print(os.getcwd())

try:
    print('읽기 --------')
    # f1 = open(r'C:\Users\acorn\Desktop\MyGitRepo\Python 기초\pro1\pack3\file_test.txt' , mode='r' , encoding = 'utf8')
    f1 = open('file_test.txt' , mode='r', encoding='utf8') # mode = 'r' , 'w' , 'a' , 'b' ...
    print(f1.read())
    f1.close()
    
    print('저장---------')
    f2 = open('file_test2.txt' , mode='w' , encoding = 'utf-8')
    f2.write('My friends\n')
    f2.write('홍길동, 나길동')
    f2.close()
    
    print('추가---------')
    f3 = open('file_test2.txt' , mode='a' , encoding='utf-8')
    f3.write('\n손오공')
    f3.write('\n팔계')
    f3.write('\n오정')
    f3.close()
    
    print('읽기 --------')
    f4 = open('file_test2.txt' , mode='r', encoding='utf8') 
    print(f4.readline())
    print(f4.readline()) # 한줄씩 읽기
    
except Exception as e:
    print('에러: ',e)