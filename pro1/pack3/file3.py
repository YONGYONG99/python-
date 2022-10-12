# 동 이름을 입력해 해당 동 관련 우편번호 및 주소 출력

try:
    dong = input('동 이름 입력:')
    # print(dong)

    
    with open('zipcode.txt' , mode='r' , encoding='euc-kr') as f:
        line=f.readline()
        # print(line)
        while line: # 자료가 있으면 True 없으면 False
            # lines = line.split('\t')
            lines = line.split(chr(9))
            # print(lines)
            if lines[3].startswith(dong):
                # print(lines)
                print('[' + lines[0] + ']' + lines[1] + ' ' + lines[2] + ' ' + lines[3] + ' ' + lines[4])
            
            line = f.readline()
except Exception as e:
    print('err : ', e)    
