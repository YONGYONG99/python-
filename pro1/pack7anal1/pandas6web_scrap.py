# 웹 문서 읽기 - scraping
# 참고) crawling : scrap, selenium
# Beautisoup을 이용 : XML , HTML 문서 처리

import requests
from bs4 import BeautifulSoup

def spider():
    baseUrl = "https://www.naver.com/index.html"
    sourceData = requests.get(baseUrl)
    print(sourceData)

    plainText = sourceData.text
    # print(plainText)
    print(type(plainText)) # <class 'str'>
    
    # BeautifulSoup
    convertData = BeautifulSoup(plainText, 'lxml') # 인코딩해온걸 디코딩했다는 맥락으로도 볼수있다.
    # print(converData)
    print(type(convertData)) # <class 'bs4.BeautifulSoup'>
    
    # atag만 잡아오기
    for atag in convertData.find_all('a'):
        href = atag.get('href')
        title = atag.string
        print(href, ' ', title)
    
    
if __name__ == '__main__':
    spider()
    
