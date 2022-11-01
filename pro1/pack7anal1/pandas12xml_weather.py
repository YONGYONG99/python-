# 기상청 제공 날씨정보 XML 자료 읽기
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd

url = "http://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp"
data = urllib.request.urlopen(url).read()

soup = BeautifulSoup(data, 'html.parser')
# print(soup)

title = soup.find('title').string
print(title)
wf = soup.find('wf').string
print(wf)

city = soup.find_all('city')
print(city)
cityData = []
for c in city:
    cityData.append(c.string)
    
df = pd.DataFrame()
df['city'] = cityData
print(df.head(3), len(df))

print()
# select method
tempMins = soup.select('location > province + city + data > tmn') # + : 다음 형제 찾기 # <data>중에 첫번째만 가져오고싶다.
tempData = []
for t in tempMins:
    tempData.append(t.string)

df['temp_min'] = tempData
df.columns = ['지역','최저기온']
print(df.head(3))

# 파일로 담아두기
df.to_csv('날씨.csv', index=False)
print()
df2 = pd.read_csv('날씨.csv')
print(df2.head(3))

print('----df 자료로 슬라이싱...------')
# iloc ...  숫자를 기반
print(df.iloc[0]) # 1차원 배열 취급을 당한다.

print(df.iloc[0:2, :])
print(df.iloc[0:2, 0:1])
print(df.iloc[0:2, 0:2])
print()
print(df['지역'][0:2]) # 지역에 대해서 0행과 1행만 나와라 , 시리즈다 (밑에 저런거 뜨면 시리즈)

# loc
print(df.loc[1:3])
print(df[1:4]) # 위랑 똑같다.
print(df.loc[[1,3]])
print(df.loc[:, '지역'].head(2)) # 지역에대한 모든행 -> 그중 2개만 나와라
print(df.loc[1:3, ['최저기온','지역']])
print(df.loc[:, '지역'][1:4])

print('---------')
df = df.astype({'최저기온':int})
print(df.info())
print(df['최저기온'].mean(), ' ', df['최저기온'].std())

print(df['최저기온'] >=6 )

print(df.loc[df['최저기온'] >=7])

print(df.sort_values(['최저기온'], ascending=True))






