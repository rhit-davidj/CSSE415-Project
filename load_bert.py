import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle # Import pickle to save the scaler
import os     # Import os to create directory

# Directory to save the final model and scaler
output_dir_final = "./best_coffee_regression_model"

def load_model_and_scaler(model_dir):
    # Load the model
    loaded_model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    loaded_tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

    # Load the scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)

    return loaded_model, loaded_tokenizer, loaded_scaler

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

# Example usage (uncomment to run):
print("\n--- Demonstrating Loading and Prediction ---")
try:
    loaded_model, loaded_tokenizer, loaded_scaler = load_model_and_scaler(output_dir_final)
    loaded_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(loaded_device)

    sample_text = ["test.", "Mild flavor, a bit bland."]
    predicted_ratings = predict_and_denormalize(sample_text, loaded_model, loaded_tokenizer, loaded_scaler, loaded_device)
    print(f"Sample text: '{sample_text[0]}'")
    print(f"Predicted Original Scale Rating: {predicted_ratings[0]:.2f}")
    print(f"Sample text: '{sample_text[1]}'")
    print(f"Predicted Original Scale Rating: {predicted_ratings[1]:.2f}")

except FileNotFoundError:
    print(f"Could not load model/scaler from {output_dir_final}. Run the training first.")
except Exception as e:
     print(f"An error occurred during loading/prediction demonstration: {e}")
