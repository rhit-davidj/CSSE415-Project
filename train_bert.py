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

# Load the dataset
df = pd.read_csv("coffee_analysis.csv")

# Drop rows with any missing values
df.dropna(inplace=True)

# Store original ratings before scaling for potential later use (optional, but good practice)
original_ratings = df['rating'].copy()

# Normalize ratings
scaler = MinMaxScaler()
# Fit the scaler *before* splitting to get the full range of original data
df['rating'] = scaler.fit_transform(df[['rating']])

# Combine text columns
df['text'] = df[['desc_1', 'desc_2', 'desc_3']].fillna('').agg(' '.join, axis=1)
df['rating'] = df['rating'].astype(float) # Ensure rating is float after scaling

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['rating'].tolist(), test_size=0.2, random_state=1
)

# Tokenization using DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

# Custom dataset
class CoffeeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Ensure the label is a tensor of type float
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = CoffeeDataset(train_encodings, train_labels)

val_dataset = CoffeeDataset(val_encodings, val_labels)


# DistilBERT for regression
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1, problem_type='regression'
)

# Training arguments
# Use a specific output directory for checkpoints saved during training
checkpoint_output_dir = "./distilbert_regression_checkpoints"
training_args = TrainingArguments(
    output_dir=checkpoint_output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    logging_dir="./logs",
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True, # This ensures the best model is loaded after training
    metric_for_best_model="eval_loss", # Metric to monitor for best model
    learning_rate=2e-5,
    fp16=torch.cuda.is_available()
)

# Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # For regression, predictions are typically a 1D array/tensor
    predictions = predictions.squeeze()
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {"mse": mse, "r2": r2}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
    compute_metrics=compute_metrics
)

# Ensure model on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
print("Starting training...")
trainer.train()
print("Training finished. Best model loaded.")

# Evaluate the BEST model on the validation set
print("Evaluating best model on validation set...")
val_eval_results = trainer.evaluate(val_dataset)
print(f"Validation Evaluation Results: {val_eval_results}")

# Evaluate the BEST model on the training set
print("Evaluating best model on training set...")
train_eval_results = trainer.evaluate(train_dataset)
print(f"Training Evaluation Results: {train_eval_results}")

# Print the desired R2 scores
print("-" * 30)
# The metrics dictionary from evaluate contains keys like 'eval_r2'
print(f"Best Model Training R2: {train_eval_results.get('eval_r2', 'N/A'):.4f}")
print(f"Best Model Validation R2: {val_eval_results.get('eval_r2', 'N/A'):.4f}")
print(f"Best Model Validation MSE: {val_eval_results.get('eval_mse', 'N/A'):.4f}")
print("-" * 30)

# --- Save the best model, tokenizer, and scaler ---

# Directory to save the final model and scaler
output_dir_final = "./best_coffee_regression_model"
os.makedirs(output_dir_final, exist_ok=True) # Create directory if it doesn't exist

print(f"Saving best model, tokenizer, and scaler to {output_dir_final}")

# Save the best model (Trainer automatically loads the best one if load_best_model_at_end=True)
trainer.model.save_pretrained(output_dir_final)

# Save the tokenizer (needed for preprocessing new text data later)
tokenizer.save_pretrained(output_dir_final)

# Save the scaler object (needed to reverse the normalization of predictions)
scaler_filename = os.path.join(output_dir_final, "scaler.pkl")
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)

print("Saving complete.")