
from time import strftime

from bs4 import BeautifulSoup

import requests
import re
import pandas as pd
import datetime

from unicodedata import category

category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']

# url ='https://news.naver.com/section/100'
#
#
# resp = requests.get(url) #요청하면 웹서버에서 응답
# print(list(resp))
# # URL 주소를 입력하면 웹서버에게 요청하면 응답하고 HTML문서를 보내줌
#
# #HTML문서로 응답
# soup = BeautifulSoup(resp.text, 'html.parser')
# print(soup)
#
# # html문서에서 헤드라인 제목
# title_tags = soup.select('.sa_text_strong')
# print(len(title_tags))
#
# for title_tag in title_tags:
#     print(title_tag.text)
# print(len(title_tags))

df_titles = pd.DataFrame()

#for i in range(6):
for i in range(4,6): # W ,I ('World', 'IT')
    url = 'https://news.naver.com/section/10{}'.format(i) # 100,101~106까지 가져오기
    resp = requests.get(url)  # 요청하면 웹서버에서 응답
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tags = soup.select('.sa_text_strong')
    titles = []
    for title_tag in title_tags:
        title = title_tag.text #문장으로 변경
        title = re.compile('[^가-힣 ]').sub('', title) # 전처리, # 한글하고 띄어쓰기 빼고 널문자 추가
        titles.append(title)
    df_section_titles = pd.DataFrame(titles, columns=['titles']) # 데이터프레임 생성
    df_section_titles['category'] = category[i] # 카테고리 라벨 붙이기
    df_titles = pd.concat([df_titles, df_section_titles], axis='rows', ignore_index=True) #빈 데이터프레임에 row정장

print(df_titles.head())
df_titles.info()
print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news_4_5_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False) # 나노second단위 받은 시간으로 오늘 날짜로 바꿔서 저장

