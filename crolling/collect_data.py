# Google Maps crolling
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import pandas as pd

# 크롬 드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless') # 크롤러가 백그라운드에서 실행되도록 설정
driver = webdriver.Chrome(options=options)

def get_google_maps_data(location, search_query):
    url = f'https://www.google.com/maps/search/{search_query}+in+{location}'
    driver.get(url)
    time.sleep(5)  # 페이지 로딩 대기
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    places = soup.find_all('div', {'class': 'section-result-content'})
    
    data = []
    for place in places:
        name = place.find('h3', {'class': 'section-result-title'}).text
        try:
            rating = place.find('span', {'class': 'section-result-rating'}).text
        except:
            rating = 'N/A'
        try:
            reviews = place.find('span', {'class': 'section-result-num-ratings'}).text
        except:
            reviews = 'N/A'
        
        data.append([name, rating, reviews])
    
    return pd.DataFrame(data, columns=['Name', 'Rating', 'Reviews'])

location = 'Seoul'
search_query = 'restaurants'
df_google_maps = get_google_maps_data(location, search_query)
print(df_google_maps)
##################################33
# Yelp crolling
import requests
import pandas as pd

API_KEY = 'your_yelp_api_key'  # Yelp API 키 입력
HEADERS = {'Authorization': f'Bearer {API_KEY}'}
SEARCH_API_URL = 'https://api.yelp.com/v3/businesses/search'

def get_yelp_data(location, term):
    params = {
        'term': term,
        'location': location,
        'limit': 50
    }
    response = requests.get(SEARCH_API_URL, headers=HEADERS, params=params)
    businesses = response.json().get('businesses')
    
    data = []
    for biz in businesses:
        name = biz['name']
        rating = biz['rating']
        review_count = biz['review_count']
        data.append([name, rating, review_count])
    
    return pd.DataFrame(data, columns=['Name', 'Rating', 'Reviews'])

location = 'Seoul'
term = 'restaurants'
df_yelp = get_yelp_data(location, term)
print(df_yelp)
