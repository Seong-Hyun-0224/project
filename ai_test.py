###############3
터미널 에러 출력 해결 요먕
(SVSAP_ML) C:\Users\SeongHyunKim\workspace\svsap\project>python ai_test.py
2024-07-30 17:09:48.955132: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-30 17:09:51.477840: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-30 17:09:56.872894: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Traceback (most recent call last):
  File "C:\Users\SeongHyunKim\workspace\svsap\project\ai_test.py", line 8, in <module>
    model = load_model('restaurant_recommendation_model.h5')
  File "C:\Users\SeongHyunKim\anaconda3\envs\SVSAP_ML\lib\site-packages\keras\src\saving\saving_api.py", line 189, in load_model
    return legacy_h5_format.load_model_from_hdf5(
  File "C:\Users\SeongHyunKim\anaconda3\envs\SVSAP_ML\lib\site-packages\keras\src\legacy\saving\legacy_h5_format.py", line 155, in load_model_from_hdf5
    **saving_utils.compile_args_from_training_config(
  File "C:\Users\SeongHyunKim\anaconda3\envs\SVSAP_ML\lib\site-packages\keras\src\legacy\saving\saving_utils.py", line 143, in compile_args_from_training_config     
    loss = _deserialize_nested_config(losses.deserialize, loss_config)
  File "C:\Users\SeongHyunKim\anaconda3\envs\SVSAP_ML\lib\site-packages\keras\src\legacy\saving\saving_utils.py", line 202, in _deserialize_nested_config
    return deserialize_fn(config)
  File "C:\Users\SeongHyunKim\anaconda3\envs\SVSAP_ML\lib\site-packages\keras\src\losses\__init__.py", line 149, in deserialize
    return serialization_lib.deserialize_keras_object(
  File "C:\Users\SeongHyunKim\anaconda3\envs\SVSAP_ML\lib\site-packages\keras\src\saving\serialization_lib.py", line 575, in deserialize_keras_object
    return deserialize_keras_object(
  File "C:\Users\SeongHyunKim\anaconda3\envs\SVSAP_ML\lib\site-packages\keras\src\saving\serialization_lib.py", line 678, in deserialize_keras_object
    return _retrieve_class_or_fn(
  File "C:\Users\SeongHyunKim\anaconda3\envs\SVSAP_ML\lib\site-packages\keras\src\saving\serialization_lib.py", line 812, in _retrieve_class_or_fn
    raise TypeError(
TypeError: Could not locate function 'mse'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`. Full object config: {'module': 'keras.metrics', 'class_name': 'function', 'config': 'mse', 'registered_name': 'mse'}


#################3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model('restaurant_recommendation_model.h5')

# 테스트할 사용자 선택
test_user_id = 0  # 예를 들어 첫 번째 사용자

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
plt.xlabel('예측 점수')
plt.ylabel('식당')
plt.title('사용자에 대한 식당 추천 결과')
plt.gca().invert_yaxis()  # 높은 점수가 위로 가도록
plt.show()
