import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense

# 데이터 로드
users = pd.read_csv('data/users.csv')
foods = pd.read_csv('data/foods.csv')
restaurants = pd.read_csv('data/restaurants.csv')
reviews = pd.read_csv('data/reviews.csv')

# 데이터 전처리
num_users = users['user_id'].nunique()
num_restaurants = restaurants['restaurant_id'].nunique()
num_foods = foods['food_id'].nunique()

# 리뷰 데이터 준비
user_ids = reviews['user_id'].values
restaurant_ids = reviews['restaurant_id'].values
food_ids = reviews['food_id'].values
ratings = reviews['rating'].values

# 데이터 분할
train_indices, test_indices = train_test_split(np.arange(len(ratings)), test_size=0.2, random_state=42)
train_user_ids, test_user_ids = user_ids[train_indices], user_ids[test_indices]
train_restaurant_ids, test_restaurant_ids = restaurant_ids[train_indices], restaurant_ids[test_indices]
train_food_ids, test_food_ids = food_ids[train_indices], food_ids[test_indices]
train_ratings, test_ratings = ratings[train_indices], ratings[test_indices]

# 임베딩 레이어 정의
user_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=num_users, output_dim=50)(user_input)
user_vec = Flatten()(user_embedding)

restaurant_input = Input(shape=(1,))
restaurant_embedding = Embedding(input_dim=num_restaurants, output_dim=50)(restaurant_input)
restaurant_vec = Flatten()(restaurant_embedding)

food_input = Input(shape=(1,))
food_embedding = Embedding(input_dim=num_foods, output_dim=50)(food_input)
food_vec = Flatten()(food_embedding)

# 임베딩 벡터 결합
concat_vec = Concatenate()([user_vec, restaurant_vec, food_vec])
dense = Dense(128, activation='relu')(concat_vec)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='linear')(dense)

# 모델 정의 및 컴파일
model = Model(inputs=[user_input, restaurant_input, food_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# 모델 학습
model.fit([train_user_ids, train_restaurant_ids, train_food_ids], train_ratings, epochs=12, batch_size=32, validation_split=0.2)

# 모델 저장
model.save('data/restaurant_recommendation_model.h5')
