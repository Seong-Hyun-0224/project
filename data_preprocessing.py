import pandas as pd

# 데이터 로드
users = pd.read_csv('data/users.csv')
foods = pd.read_csv('data/foods.csv')
restaurants = pd.read_csv('data/restaurants.csv')
reviews = pd.read_csv('data/reviews.csv')

print(users.head())
print(foods.head())
print(restaurants.head())
print(reviews.head())
print("\n")
print("*************data_crolling & preprocessing complete.****************")