import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# 데이터 로드
users = pd.read_csv('data/users.csv')
foods = pd.read_csv('data/foods.csv')
restaurants = pd.read_csv('data/restaurants.csv')
reviews = pd.read_csv('data/reviews.csv')

# 모델 로드
# model = load_model('data/restaurant_recommendation_model.h5')
model = load_model('data/restaurant_recommendation_model.h5', custom_objects={'mse': MeanSquaredError()})

# 데이터 전처리
num_users = users['user_id'].nunique()
num_restaurants = restaurants['restaurant_id'].nunique()
num_foods = foods['food_id'].nunique()

# 리뷰 데이터 준비
user_ids = reviews['user_id'].values
restaurant_ids = reviews['restaurant_id'].values
food_ids = reviews['food_id'].values
ratings = reviews['rating'].values

# 특정 사용자에 대한 예측
test_user_id = 0  # 예시로 첫 번째 사용자
user_country = users.loc[users['user_id'] == test_user_id, 'country'].values[0]

# 해당 사용자의 국가별 리뷰 데이터 선택
country_reviews = reviews[reviews['country'] == user_country]

# 사용자의 국가별 선호도를 반영한 식당 및 메뉴 추천
test_user_ids = np.array([test_user_id] * len(country_reviews))
test_restaurant_ids = country_reviews['restaurant_id'].values
test_food_ids = country_reviews['food_id'].values

# 예측 점수 계산
predictions = model.predict([test_user_ids, test_restaurant_ids, test_food_ids])

# 예측 점수 정렬
sorted_indices = np.argsort(predictions[:, 0])[::-1]
sorted_restaurant_ids = test_restaurant_ids[sorted_indices]
sorted_food_ids = test_food_ids[sorted_indices]
sorted_predictions = predictions[sorted_indices]

# 예측 점수를 5점 만점으로 정규화하고 퍼센트로 변환
max_prediction = np.max(sorted_predictions)
min_prediction = np.min(sorted_predictions)
normalized_predictions = (sorted_predictions - min_prediction) / (max_prediction - min_prediction) * 5
normalized_predictions_percent = normalized_predictions / 5 * 100

# 추천 결과 출력
recommended_restaurants = restaurants.loc[restaurants['restaurant_id'].isin(sorted_restaurant_ids)]
recommended_foods = foods.loc[foods['food_id'].isin(sorted_food_ids)]
print("Recommended Restaurants and Foods:")
for i in range(10):
    print(f"{recommended_restaurants.iloc[i]['restaurant_name']} - {recommended_foods.iloc[i]['food_name']}: {normalized_predictions_percent[i, 0]:.2f}%")

# 추천 결과 시각화
plt.figure(figsize=(12, 8))
plt.barh(np.arange(10), normalized_predictions_percent[:10, 0], align='center')
plt.yticks(np.arange(10), [f"{recommended_restaurants.iloc[i]['restaurant_name']} - {recommended_foods.iloc[i]['food_name']}" for i in range(10)])
plt.xlabel('Recommendation Score (%)')
plt.title('Top 10 Restaurant and Food Recommendations')
plt.gca().invert_yaxis()

# 그래프를 파일로 저장
plt.savefig('data/recommendation_results.png')

# 그래프 표시
plt.show()
