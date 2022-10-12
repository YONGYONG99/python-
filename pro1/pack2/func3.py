jumsu = [95, 70 , 60, 50 ,100]
number = 0
for jum in jumsu:
    number +=1
    if jum < 70:continue # 이거 하면 밑으로 안내려가겠지?
    print('%d번째 수험생은 합격'%number)