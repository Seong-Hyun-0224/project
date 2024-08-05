def preprocess_data(df):
    df['Rating'] = df['Rating'].replace('N/A', 0).astype(float)
    df['Reviews'] = df['Reviews'].str.replace(' reviews', '').replace('N/A', 0).astype(int)
    df.dropna(inplace=True)
    return df

df_google_maps_cleaned = preprocess_data(df_google_maps)
df_yelp_cleaned = preprocess_data(df_yelp)

# 데이터 합치기
df_combined = pd.concat([df_google_maps_cleaned, df_yelp_cleaned], ignore_index=True)
print(df_combined)




