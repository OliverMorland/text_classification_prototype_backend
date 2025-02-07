import os
from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load DistilBERT multilingual classifier
classifier = pipeline("zero-shot-classification", model="distilbert-base-multilingual-cased")

# Define possible categories
CATEGORIES = ["website performance", "website layout", "support staff"]

@app.route('/')
def home():
    return "Flask Backend is Running!"

@app.route('/classify', methods=['POST'])
def classify_feedback():
    data = request.json
    feedback_list = data.get("feedback", [])
    results = []
    
    # for feedback in feedback_list:
    #     classification = classifier(feedback, CATEGORIES)
    #     top_category = classification['labels'][0]  # Select the highest confidence category
    #     results.append({"feedback": feedback, "category": top_category})
    # Test response to check if speed issue is model-related
    for feedback in feedback_list:
        results.append({"feedback": feedback, "category": "test-category"})
    
    return jsonify({"classified": results})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
