import os
from data_loader import load_data
from preprocess import run_eda, preprocess
from model import train_and_evaluate

os.makedirs('plots', exist_ok=True)

print("Loading data...")
df = load_data()
print(f"Dataset: {df.shape[0]} reviews loaded")

print("\n--- EDA ---")
run_eda(df)

print("\n--- Preprocessing ---")
X, y, tokenizer, le = preprocess(df)
print(f"X shape: {X.shape}, Classes: {le.classes_}")

print("\n--- Model Training ---")
model = train_and_evaluate(X, y)

print("\nDone! Check /plots folder for all charts.")