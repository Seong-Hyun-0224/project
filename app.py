from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 모델 로드
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)
    return jsonify({'rating': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
