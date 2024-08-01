import pandas as pd
import numpy as np

# 설정
num_users = 100
num_restaurants = 50
num_foods = 200
num_reviews = 1000

# 사용자 데이터 생성
np.random.seed(42)
user_ids = np.arange(num_users)
countries = ['USA', 'Korea', 'Japan']
preferred_flavors = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy']
disliked_ingredients = ['Meat', 'Seafood', 'Vegetable']
preferred_cuisines = ['Japanese', 'Korean', 'American', 'Indian']
allergy_ingredients = ['peanut', 'nut', 'shellfish', 'milk', 'egg', 'wheat']

users = pd.DataFrame({
    'user_id': user_ids,
    'country': np.random.choice(countries, num_users),
    'preferred_flavors': np.random.choice(preferred_flavors, num_users),
    'non_preferred_flavors': np.random.choice(preferred_flavors, num_users),
    'disliked_ingredients': np.random.choice(disliked_ingredients, num_users),
    'preferred_cuisines': np.random.choice(preferred_cuisines, num_users),
    'allergy_ingredients': np.random.choice(allergy_ingredients, num_users)
})

# 음식 데이터 생성
food_ids = np.arange(num_foods)
foods = pd.DataFrame({
    'food_id': food_ids,
    'food_name': [f'Food_{i}' for i in food_ids],
    'basic_taste': np.random.choice(preferred_flavors, num_foods),
    'other_tastes': [','.join(np.random.choice(preferred_flavors, 3, replace=False)) for _ in range(num_foods)],
    'ingredients': [','.join(np.random.choice(disliked_ingredients, 2, replace=False)) for _ in range(num_foods)]
})

# 식당 데이터 생성
restaurant_ids = np.arange(num_restaurants)
restaurants = pd.DataFrame({
    'restaurant_id': restaurant_ids,
    'restaurant_name': [f'Restaurant_{i}' for i in restaurant_ids],
    'food_ids': [','.join(map(str, np.random.choice(food_ids, np.random.randint(1, 10), replace=False))) for _ in range(num_restaurants)]
})

# 주관적인 리뷰 텍스트 생성
review_texts_positive = [
    "Absolutely loved it! The flavors were perfectly balanced.",
    "One of the best meals I've ever had. Highly recommend!",
    "Fantastic food and great service. Will definitely come back.",
    "Delicious and delightful! Every bite was amazing.",
    "A culinary masterpiece. The chef knows what they're doing!"
]

review_texts_negative = [
    "Terrible experience. The food was bland and tasteless.",
    "Not worth the money. Very disappointed.",
    "Service was slow and the food was overcooked.",
    "Wouldn't recommend it to anyone. Really bad.",
    "Awful. The worst dining experience I've had in a while."
]

# 리뷰 데이터 생성
review_user_ids = np.random.choice(user_ids, num_reviews)
review_restaurant_ids = np.random.choice(restaurant_ids, num_reviews)
review_food_ids = [np.random.choice(list(map(int, restaurants.loc[restaurants['restaurant_id'] == rid, 'food_ids'].values[0].split(',')))) for rid in review_restaurant_ids]
ratings = np.random.randint(1, 6, num_reviews)
review_texts = [
    review_texts_positive[rating - 4] if rating > 3 else review_texts_negative[rating - 1]
    for rating in ratings
]

reviews = pd.DataFrame({
    'user_id': review_user_ids,
    'restaurant_id': review_restaurant_ids,
    'food_id': review_food_ids,
    'rating': ratings,
    'review_text': review_texts,
    'country': [users.loc[users['user_id'] == uid, 'country'].values[0] for uid in review_user_ids]
})

# 데이터 저장
users.to_csv('data/users.csv', index=False)
foods.to_csv('data/foods.csv', index=False)
restaurants.to_csv('data/restaurants.csv', index=False)
reviews.to_csv('data/reviews.csv', index=False)
