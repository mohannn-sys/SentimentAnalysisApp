from flask import Flask, render_template, request
from transformers import pipeline
import pickle

app = Flask(__name__)

# Loading pickled BERT model
model_path = "sentiment_model.pkl"
classifier = pickle.load(open(model_path, "rb"))

# Function to predict sentiment using the model
def predict_sentiment(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    if label == "POSITIVE":
        return (score, "positive")
    else:
        return (score, "negative")

# Defining the route for the home/index page
@app.route("/")
def index():
    return render_template("index.html")

# Defining the route for the result page
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    score, sentiment = predict_sentiment(text)
    probability = round(float(score) * 100, 2)
    return render_template('result.html', text=text, sentiment=sentiment, probability=probability)


if __name__ == "__main__":
    app.run(debug=True)
