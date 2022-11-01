# 다음날 안될수 있음 영원하지 않음

# 웹문서 읽기1
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup

print('벅스 차트 출력하기 ---')
url = urlopen("https://music.bugs.co.kr/chart")
soup = BeautifulSoup(url.read(), 'html.parser')
#print(soup)
musics = soup.find_all('td', class_='check')
print(musics)

for i, music in enumerate(musics):
    print("{}위:{}".format(i +1 , music.input['title']))

print('------------------------')

# 웹 문서 읽기2
import urllib.request as req
url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"
wiki = req.urlopen(url)
print(wiki)
soup2 = BeautifulSoup(wiki, 'html.parser')
#print(soup2)
#mw-content-text > div.mw-parser-output > p:nth-child(6)   ->  웹페이지 검사에서 copy selector
print(soup2.select("div.mw-parser-output > p > b"))
result = soup2.select("div.mw-parser-output > p > b")

for a in result:
    # print(a.string)
    if(a.string !=None):
        print(a.string)

print('------------------------')

# 웹 문서 읽기3 - daum의 뉴스 정보 읽기
url = "https://news.daum.net/society#1"
daum = req.urlopen(url)

soup3 = BeautifulSoup(daum, 'lxml') # lxml써보기
#gnbContent > div > ul > li:nth-child(1) > a > span
print(soup3.select_one("div > strong > a"))
data = soup3.select_one("div > strong > a")
for i in data:
    print(i)

print()
datas = soup3.select("div > strong > a")

for i in datas[:5]:
    print(i)

print()
datas2 = soup3.findAll("a")
# print(datas2[:5])
for i in datas2[10:15]:  # 5개만 가져오기
    # print(i)
    h = i.attrs['href']
    t = i.string
    print('href:%s text:%s'%(h,t))





