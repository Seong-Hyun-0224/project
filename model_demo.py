# 데이터 생성
# 임의의 식당 리뷰 가상 데이터 생성

import pandas as pd
import numpy as np

# 식당 10개와 사용자 20명
restaurant_ids = [f"restaurant_{i}" for i in range(1, 11)]
user_ids = [f"user_{i}" for i in range(1, 21)]

# 임의로 리뷰 데이터 생성
np.random.seed(42)
data = {
    "user_id": np.random.choice(user_ids, 100),
    "restaurant_id": np.random.choice(restaurant_ids, 100),
    "rating": np.random.randint(1, 6, 100),
    "review": ["Sample review text" for _ in range(100)]  # 임의의 리뷰 텍스트
}

reviews_df = pd.DataFrame(data)
reviews_df.to_csv("reviews.csv", index=False)
print(reviews_df.head())

#############################
# 2. data preprocessing
reviews_df = pd.read_csv("reviews.csv")

# 사용자-식당 평점 행렬 생성
rating_matrix = reviews_df.pivot_table(values="rating", index="user_id", columns="restaurant_id")

# 결측치(NaN)를 0으로 채우기
rating_matrix.fillna(0, inplace=True)

################################
# 모델 학습
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Surprise 라이브러리에서 데이터 로드
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(reviews_df[["user_id", "restaurant_id", "rating"]], reader)

# 학습 및 테스트 데이터셋 분할
trainset, testset = train_test_split(data, test_size=0.2)

# SVD 모델 학습
model = SVD()
model.fit(trainset)

# 테스트 데이터셋으로 예측
predictions = model.test(testset)
accuracy.rmse(predictions)

##################################
# 추천 시스템 빌드
def get_top_n_recommendations(model, user_id, n=5):
    user_ratings = rating_matrix.loc[user_id]
    unseen_restaurants = user_ratings[user_ratings == 0].index.tolist()
    predictions = [model.predict(user_id, restaurant_id).est for restaurant_id in unseen_restaurants]

    recommended_restaurants = sorted(zip(unseen_restaurants, predictions), key=lambda x: x[1], reverse=True)[:n]
    return recommended_restaurants

# 특정 사용자에게 추천
user_id = "user_1"
recommendations = get_top_n_recommendations(model, user_id, n=5)
print(f"Top 5 recommendations for {user_id}: {recommendations}")


########################################
# 시나리오 기반 추천 시스템 테스트
user_preferences = {
    "less_salty": True,
    "no_beans": True,
    "nut_allergy": True
}

# 사용자의 선호도를 반영한 추천 함수
def personalized_recommendations(model, user_id, preferences, n=5):
    recommendations = get_top_n_recommendations(model, user_id, n)
    # 추가 필터링 로직 적용 (예: 사용자가 싱겁게 먹는 것을 선호하는 경우)
    filtered_recommendations = [rec for rec in recommendations if satisfies_preferences(rec, preferences)]
    return filtered_recommendations

def satisfies_preferences(recommendation, preferences):
    # 실제 식당 정보와 비교하여 조건을 만족하는지 확인
    # 여기서는 임의로 모든 추천을 만족하는 것으로 가정(이부분은 실제로 서비스가 구축되고 제공되면서 들어오는 피드백에 따라 발전될 예정)
    return True

# 특정 사용자에게 시나리오 기반 추천
personalized_recs = personalized_recommendations(model, user_id, user_preferences, n=5)
print(f"Personalized recommendations for {user_id}: {personalized_recs}")
