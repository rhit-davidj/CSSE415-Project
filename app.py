from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# load dataset
df = pd.read_csv('coffee_analysis.csv')

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# display main page
@app.route('/')
def index():
    return render_template('index.html')

# calculate coffee predictions
@app.route('/predict', methods=['POST'])
def predict():
    similar_coffees_list = []
    similar_coffees_list.append({
        'name': "test",
        'notes': "placeholder notes", # Or other relevant info
        'similarity_score': 1
    })
    
    predicted_rating = 100.0
    processed_notes = "testing notes"
    return jsonify({
        'predicted_rating': predicted_rating,
        'similar_coffees': similar_coffees_list,
        'processed_input': processed_notes
    })

if __name__ == '__main__':
    app.run(debug=True)