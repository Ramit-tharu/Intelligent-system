from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data.get('review')
    if not review:
        return jsonify({'error': 'No review provided'}), 400
    # Preprocess and predict
    transformed_review = vectorizer.transform([review])
    prediction = model.predict(transformed_review)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

