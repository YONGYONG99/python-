# 네이버 제공 코스피 정보를 읽어 csv 파일로 저장

import csv
import requests
from bs4 import BeautifulSoup

url = "https://finance.naver.com/sise/sise_market_sum.naver?&page={}"
fname = "네이버_코스피.csv"
# fObj = open(fname, mode='w', encoding='UTF-8', newline='') # 공백행 제외
fObj = open(fname, mode='w', encoding='UTF-8-sig', newline='') # 엑셀에서 읽을 때 한글 깨짐

writer = csv.writer(fObj) 
title = "N    종목명    현재가    전일비    등락률    액면가    시가총액    상장주식수    외국인비율    거래량    PER    ROE".split() # 공백을 기준으로 짜르기
print(title)
writer.writerow(title)

for page in range(1,3):
    res = requests.get(url.format(str(page)))
    # print(res)
    res.raise_for_status() # 읽기 실패하면 작업 중지
    soup = BeautifulSoup(res.text , 'html.parser')
    # print(soup)
    datas = soup.find("table", attrs={"class":"type_2"}).find("tbody").find_all("tr")
    # print(datas)
    
    for row in datas:
        cols = row.findAll("td")
        if len(cols) <= 1:continue  # [''] 해결
        data = [col.get_text().strip() for col in cols] # strip() : '\n\n\t\t\t\t\n\n\n' 이런거 제거시킴
        # print(data)
        writer.writerow(data)
        
        
fObj.close()

# csv 읽기
import pandas as pd
import numpy as np
df = pd.read_csv(fname)
print(df.head(5))                                                                                                                                                                                                                                                


print("\n 네이버 웹툰 순위")

from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
url = urlopen("https://comic.naver.com/index")
soup = BeautifulSoup(url.read(),'html.parser')
# print(soup)

webtoons = soup.find_all('ol', class_='asideBoxRank')
print(webtoons)
print()
for i, webtoon in enumerate(webtoons):
    print("{}위:{}".format(i +1 , webtoon.input['title']))








