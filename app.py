from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification 
import torch
import pickle
import os
model_dir = "./best_model"

app = Flask(__name__)

# load dataset
df = pd.read_csv('coffee_analysis.csv')

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = df["desc_1"].tolist()
embeddings = embedding_model.encode(sentences)
df["embedding"] = embeddings.tolist()

# Load the bert model
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

# Load the scaler
scaler_path = os.path.join(model_dir, "scaler.pkl")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

def predict_and_denormalize(text, model, tokenizer, scaler, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # The output is typically a tuple, the first element is the logits/predictions
        # For regression with num_labels=1, this will be a tensor like [[prediction]]
        predictions_scaled = outputs.logits.squeeze().cpu().numpy().reshape(-1, 1) # Ensure shape is (n_samples, 1)

    # Denormalize the prediction
    predictions_original_scale = scaler.inverse_transform(predictions_scaled)

    # If processing a single text, return the single prediction
    return predictions_original_scale[0][0] if len(text) == 1 else predictions_original_scale.flatten().tolist()

# display main page
@app.route('/')
def index():
    return render_template('index.html')

# calculate coffee predictions
@app.route('/predict', methods=['POST'])
def predict():           
    data = request.get_json()
    notes = data.get('notes', '')
    print("Received notes:", notes)

    # Check if notes is empty
    if not notes:
        return jsonify({'error': 'No notes provided'}), 400
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # predicct the rating
    predicted_rating = predict_and_denormalize(notes, model, tokenizer, scaler, device)[0]
    predicted_rating = np.round(predicted_rating)
    print("Predicted rating:", predicted_rating)

    # create embeddings for the notes
    embedding = embedding_model.encode([notes])
    # calculate cosine similarity
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(embedding, np.array(x).reshape(1, -1))[0][0])
    # sort by similarity
    df_sorted = df.sort_values(by='similarity', ascending=False)
    # get the top 5 similar coffees
    similar_coffees_list = []
    for coffee in df_sorted.head(5).itertuples():
        similar_coffees_list.append({
            'name': coffee.name,
            'similarity': np.round(coffee.similarity*100),
            'notes': coffee.desc_1,
            'rating': coffee.rating,
        })

    return jsonify({
        'predicted_rating': predicted_rating,
        'similar_coffees': similar_coffees_list,
    })

if __name__ == '__main__':
    app.run(debug=True)