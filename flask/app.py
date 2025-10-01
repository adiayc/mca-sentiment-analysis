# app.py

import matplotlib
from collections import Counter

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NEW: NLTK Download (Run this once or ensure it's run during deployment) ---
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
# --- END NEW NLTK Download ---

import matplotlib.dates as mdates


app = Flask(__name__)
CORS(
    app, resources={r"/*": {"origins": "http://localhost:5173"}}
)  # Enable CORS for all routes


# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r"\n", " ", comment)
        
        # Remove URLs (often irrelevant for word clouds)
        comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
        
         # Remove user mentions and hashtags (often just noise for general word clouds)
        comment = re.sub(r'@\w+|#\w+', '', comment)
        
        comment = re.sub(r'[^\w\s]', '', comment) # Keep only alphanumeric and whitespace
        
        stop_words = set(stopwords.words("english"))
        
        common_stopwords = set(stopwords.words('english'))


        additional_stopwords = {'mca', 'v3', 'act', 'rule', 'portal', 'form', 'company', 'companies', 'ministry', 'director', 'board', 'section', 'code', 'business', 'process', 'filing', 'compliance', 'please', 'due', 'issue', 'get', 'make', 'may', 'see', 'use', 'many', 'much', 'also', 'one', 'much', 'say', 'find', 'take', 'need', 'still', 'like', 'even', 'want', 'get', 'go', 'give', 'come', 'would', 'could', 'must', 'should', 'got', 'has', 'have', 'had', 'been', 'was', 'were', 'being', 'am', 'is', 'are', 'be', 'do', 'does', 'did', 'done', 'doing', 'shall', 'will', 'may', 'might', 'can', 'could', 'ought', 'would', 'must', 'need'}
        
        stop_words_for_wordcloud = common_stopwords.union(additional_stopwords)
        
        comment = " ".join([word for word in comment.split() if word not in stop_words_for_wordcloud and len(word) > 1]) # Also remove single-character words

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# Load the model and vectorizer from local storage.
def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


model, vectorizer = load_model_and_vectorizer(
    ".\\models\\lightgbm_model.pkl", ".\\models\\vectorizer.pkl"
)


@app.route("/")
def home():
    return "Welcome to our flask api"


@app.route("/fetch_replies", methods=["POST"])
def fetch_replies():
    """Proxy endpoint to fetch X/Twitter replies"""
    try:
        data = request.get_json()
        post_id = data.get("post_id")
        bearer_token = data.get("bearer_token")  # Or use environment variable

        if not post_id:
            return jsonify({"error": "Post ID is required"}), 400

        if not bearer_token:
            return jsonify({"error": "Bearer token is required"}), 400

        # Make request to X API from server side
        search_query = f"conversation_id:{post_id} is:reply"
        search_url = f"https://api.twitter.com/2/tweets/search/recent"

        params = {
            "query": search_query,
            "max_results": 50,
            "tweet.fields": "text,author_id,created_at",
        }

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }

        import requests

        response = requests.get(search_url, headers=headers, params=params)

        if not response.ok:
            return (
                jsonify(
                    {"error": f"X API Error: {response.status_code} - {response.text}"}
                ),
                response.status_code,
            )

        api_data = response.json()

        if not api_data.get("data"):
            return jsonify({"error": "No replies found for this post"}), 404

        # Extract just the text from replies
        replies = [tweet["text"] for tweet in api_data["data"]]

        return jsonify({"replies": replies})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comments = data.get("comments")
    print("🔍 Received comments:", comments)

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list

        # Convert predictions to strings for consistency
        predictions = [str(int(pred)) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Return the response with original comments and predicted sentiments
    response = [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)


@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        """
        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        # if sum(sizes) == 0:
        #     raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
        """
        response = {
            "labels": ["Positive", "Neutral", "Negative"],
            "values": [
                int(sentiment_counts.get("1", 0)),
                int(sentiment_counts.get("0", 0)),
                int(sentiment_counts.get("-1", 0)),
            ],
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get(
            "comments",
        )

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = " ".join(preprocessed_comments)

        # 1. Split text into words and remove any empty strings
        words = [word for word in text.split() if word]

        # 2. Count the frequency of each word
        # We'll take the 100 most common words to avoid a cluttered cloud
        word_counts = Counter(words).most_common(100)

        # 3. Format the data for the react-d3-cloud library
        # The required format is an array of objects: [{ text: 'word', value: count }]
        word_data = [{"text": word, "value": count} for word, count in word_counts]

        # 4. Return the data as JSON
        return jsonify({"words": word_data})

    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
