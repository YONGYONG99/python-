# 일정 시간 마다 웹 문서 읽기
# import schedule # pip install schedule 스케쥴러 모듈 지원
import time # 대신 이거 쓰자
import datetime
import urllib.request as req
from bs4 import BeautifulSoup
import requests

def work():
    url = "https://finance.naver.com/marketindex/"
    # data = req.urlopen(url)             # 읽는 방법1 : 데이터를 보낼때 인코딩하여 바이너리 형태로 보낸다.
    data = requests.get(url).text         # 방법2 : 데이터를 보낼때 딕셔너리 형태로 보낸다.
    
    soup = BeautifulSoup(data, 'html.parser')
    #print(soup)
    price = soup.select_one("div.head_info > span.value").string
    print('미국USD : ', price) # 일단 정보 잘 가져옴
    
    # 5초에 한번씩 찍어보기
    t = datetime.datetime.now()
    # print(t)
    fname =  "./usd/" + t.strftime('%Y-%m-$d-$H-$M-%S') + '.txt'
    # print(fname) ./usd/2022-11-$d-$H-$M-47.txt
    
    with open(fname, 'w') as f:
        f.write(price)
        
while True:
    work()
    time.sleep(5)

work()
