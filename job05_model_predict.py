import pandas as pd  # pandas 라이브러리 임포트
import numpy as np  # numpy 라이브러리 임포트
from sklearn.model_selection import train_test_split  # 데이터 분할을 위한 함수 임포트
from sklearn.preprocessing import LabelEncoder  # 레이블 인코딩을 위한 클래스 임포트
from keras.utils import to_categorical  # 원-핫 인코딩을 위한 함수 임포트
import pickle  # 객체 직렬화를 위한 pickle 모듈 임포트
from konlpy.tag import Okt  # 한국어 형태소 분석을 위한 Okt 클래스 임포트
from tensorflow.keras.preprocessing.text import Tokenizer  # 텍스트 전처리를 위한 Tokenizer 클래스 임포트
from tensorflow.keras.preprocessing.sequence import pad_sequences  # 시퀀스 패딩을 위한 함수 임포트
from keras.models import load_model

import pandas as pd  # pandas 라이브러리 임포트
import numpy as np  # numpy 라이브러리 임포트
from sklearn.model_selection import train_test_split  # 데이터 분할을 위한 함수 임포트
from sklearn.preprocessing import LabelEncoder  # 레이블 인코딩을 위한 클래스 임포트
from keras.utils import to_categorical  # 원-핫 인코딩을 위한 함수 임포트
import pickle  # 객체 직렬화를 위한 pickle 모듈 임포트
from konlpy.tag import Okt  # 한국어 형태소 분석을 위한 Okt 클래스 임포트
from tensorflow.keras.preprocessing.text import Tokenizer  # 텍스트 전처리를 위한 Tokenizer 클래스 임포트
from tensorflow.keras.preprocessing.sequence import pad_sequences  # 시퀀스 패딩을 위한 함수 임포트
from keras.models import load_model

# CSV 파일에서 데이터프레임 읽기
df = pd.read_csv('./crawling_data/naver_headline_news_20241223.csv')
# df.drop_duplicates(inplace=True)      # 중복제거 (주석 처리됨)
df.reset_index(drop=True, inplace=True)  # 인덱스를 재설정하여 연속적인 인덱스 부여
print(df.head())  # 데이터프레임의 처음 5행 출력
df.info()  # 데이터프레임의 정보 출력
print(df.category.value_counts())  # 각 카테고리의 개수 출력

X = df['titles']  # 뉴스 제목을 X에 저장
Y = df['category']  # 카테고리를 Y에 저장

# 인코더 객체를 pickle로 저장
with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

label = encoder.classes_  # 인코딩된 레이블의 클래스 이름 저장
print(label)  # 클래스 이름 출력

labeled_y = encoder.transform(Y)  # 레이블을 인코딩하여 labeled_y에 저장
onehot_Y = to_categorical(labeled_y)  # 레이블을 원-핫 인코딩
print(onehot_Y)  # 원-핫 인코딩 결과 출력

okt = Okt()

# 각 뉴스 제목에 대해 형태소 분석 수행
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)  # 형태소 분석 후 어간 추출
print(X)  # 형태소 분석 결과 출력

# 불용어 리스트 읽기
stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)
print(stopwords)  # 불용어 리스트 출력

# 불용어 제거
for sentence in range(len(X)):
    words = []  # 불용어가 제거된 단어를 저장할 리스트
    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:  # 단어 길이가 1보다 큰 경우
            if X[sentence][word] not in list(stopwords['stopword']):  # 불용어가 아닌 경우
                words.append(X[sentence][word])  # 단어 추가
    X[sentence] = ' '.join(words)  # 단어 리스트를 문자열로 변환하여 저장

print(X[:5])  # 불용어 제거 후 첫 5개 뉴스 제목 출력

with open('./models/news_token.pickle', 'rb') as f:
  token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)  # 텍스트를 시퀀스로 변환하여 tokened_X에 저장

for i in range(len(tokened_X)):
  if len(tokened_X[i]) > 16:
    tokened_X[i] = tokened_X[i][:16]
X_pad = pad_sequences(tokened_X, 16)

print(tokened_X[0:5])  # 첫 5개 시퀀스 출력

print(X_pad[:5])  # 패딩된 시퀀스 출력

model = load_model('./models/news_category_classfication_model_0.6041055917739868.h5')
preds = model.predict(X_pad)

predicts = []
for pred in preds:
  most = label[np.argmax(pred)]
  pred[np.argmax(pred)] = 0
  second = label[np.argmax(pred)]
  predicts.append([most, second])
df['predict'] = predicts

print(df.head(30))

score = model.evaluate(X_pad, onehot_Y)
print(score[1])

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 1
print(df.OX.mean())


