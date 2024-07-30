import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate

# 데이터 로드
df = pd.read_csv('restaurant_reviews_detailed.csv')

# 사용자와 식당 ID를 숫자로 변환
user_ids = df['user'].astype('category').cat.codes.values
restaurant_ids = df['restaurant'].astype('category').cat.codes.values

# 사용자와 식당의 수
num_users = df['user'].nunique()
num_restaurants = df['restaurant'].nunique()

# 레이팅과 리뷰
ratings = df['rating'].values

# 임베딩 차원
embedding_dim = 50

# 사용자 임베딩
user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
user_vec = Flatten(name='flatten_user')(user_embedding)

# 식당 임베딩
restaurant_input = Input(shape=(1,), name='restaurant_input')
restaurant_embedding = Embedding(input_dim=num_restaurants, output_dim=embedding_dim, name='restaurant_embedding')(restaurant_input)
restaurant_vec = Flatten(name='flatten_restaurant')(restaurant_embedding)

# 임베딩 병합
concat = Concatenate()([user_vec, restaurant_vec])

# DNN 레이어
dense = Dense(128, activation='relu')(concat)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='linear')(dense)

# 모델 컴파일
model = Model([user_input, restaurant_input], output)
model.compile(optimizer='adam', loss='mse')

# 데이터 준비
x = [user_ids, restaurant_ids]
y = ratings

# 모델 훈련
model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2)

# 모델 저장
model.save('restaurant_recommendation_model.h5')
