import pandas as pd
import random

# 식당 목록
restaurants = ['Restaurant_A', 'Restaurant_B', 'Restaurant_C', 'Restaurant_D', 'Restaurant_E', 
               'Restaurant_F', 'Restaurant_G', 'Restaurant_H', 'Restaurant_I', 'Restaurant_J']

# 사용자 ID 목록
users = [f'user_{i}' for i in range(1, 201)]  # 총 200명의 사용자

# 리뷰 내용 템플릿
review_templates = [
    "This place was amazing! The food was {adj1} and {adj2}.",
    "I didn't like the food here. It was too {adj1} for my taste.",
    "The atmosphere was nice, but the food was {adj1} and {adj2}.",
    "Highly recommend this place! The food was {adj1} and the service was {adj2}.",
    "Not coming back here. The food was {adj1} and the place was {adj2}.",
    "Decent place. The food was {adj1}, but the {adj2} service could be better.",
    "The food here is {adj1} and {adj2}. Loved it!",
    "Terrible experience. The food was {adj1} and {adj2}.",
    "One of the best places I've been to! The food was {adj1} and {adj2}.",
    "The food was {adj1}, but the {adj2} ambiance made up for it."
]

# 주관적 평가 목록
adjectives1 = ['spicy', 'salty', 'bland', 'mild', 'bitter']
adjectives2 = ['great', 'terrible', 'average', 'excellent', 'poor']

# 랜덤 리뷰 생성
reviews = []
for _ in range(1000):
    user = random.choice(users)
    restaurant = random.choice(restaurants)
    rating = random.randint(1, 5)
    template = random.choice(review_templates)
    adj1 = random.choice(adjectives1)
    adj2 = random.choice(adjectives2)
    review = template.format(adj1=adj1, adj2=adj2)
    reviews.append((user, restaurant, rating, review))

# 데이터프레임 생성
df = pd.DataFrame(reviews, columns=['user', 'restaurant', 'rating', 'review'])

# CSV 파일로 저장
df.to_csv('restaurant_reviews_detailed.csv', index=False)

# 데이터 출력 확인
print(df.head())
