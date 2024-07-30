import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# 데이터프레임 로드 (예시로 CSV 파일에서 로드)
df = pd.read_csv('restaurant_reviews_detailed.csv')

# 식당 수 계산
num_restaurants = df['restaurant'].nunique()

# 사용자 정의 객체를 지정하여 모델 로드
model = load_model('restaurant_recommendation_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# 테스트할 사용자 선택
test_user_id = 0  # 예를 들어 첫 번째 사용자(이 사람이 우리 제품의 사용자 1명이라고 가정)

# 모든 식당에 대한 예측 점수 계산
test_user_ids = np.array([test_user_id] * num_restaurants)
test_restaurant_ids = np.array(range(num_restaurants))

# 예측 점수 계산
predictions = model.predict([test_user_ids, test_restaurant_ids])

# 예측 점수 정렬
sorted_indices = np.argsort(predictions[:, 0])[::-1]
sorted_restaurant_ids = test_restaurant_ids[sorted_indices]
sorted_predictions = predictions[sorted_indices]

# 식당 이름과 예측 점수 매핑
restaurant_names = df['restaurant'].astype('category').cat.categories
sorted_restaurant_names = restaurant_names[sorted_restaurant_ids]

# 추천 결과 출력
print("추천 식당:")
for name, score in zip(sorted_restaurant_names[:10], sorted_predictions[:10]):
    print(f"{name}: {score[0]}")

# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.barh(sorted_restaurant_names[:10], sorted_predictions[:10, 0], color='skyblue')
plt.xlabel('prediction score')
plt.ylabel('restaurants')
plt.title('rocommandation result for user')
plt.gca().invert_yaxis()  # 높은 점수가 위로 가도록
plt.show()
