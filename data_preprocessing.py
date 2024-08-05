import pandas as pd

# 데이터 로드
users = pd.read_csv('data/users.csv')
foods = pd.read_csv('data/foods.csv')
restaurants = pd.read_csv('data/restaurants.csv')
reviews = pd.read_csv('data/reviews.csv')

print("[user_settings dataset]")
print(users.head())
print("[foods ingrident dataset]")
print(foods.head())
print("[restaurants dataset]")
print(restaurants.head())
print("[review-combined data]")
print(reviews.head())
print("\n")
print("*************data_crolling & preprocessing complete.****************")