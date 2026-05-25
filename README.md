# Sentiment Analysis on Product Reviews

Classifies text reviews as Positive, Neutral, or Negative
using a Deep Learning LSTM model built with TensorFlow and Keras.
## Live Demo
[Streamlit link](https://customer-churn-prediction-app-ewvre3f8ipcstgjfm4jufy.streamlit.app/)
## Tech Stack
Python, TensorFlow, Keras, Pandas, NumPy,
NLTK, Scikit-learn, Matplotlib, Seaborn

## Model Architecture
- Embedding Layer (64 dimensions)
- SpatialDropout Layer (0.2)
- LSTM Layer (64 units) with Dropout
- Dense Layer (32 units, ReLU)
- Output Layer (Softmax — 3 classes)

## Dataset
Twitter US Airline Sentiment — 6,000+ real text reviews
Labels: Positive, Neutral, Negative

## Results
- Test Accuracy: ~78–82%
- Evaluated using Confusion Matrix and Classification Report

## Project Steps
1. Load and explore sentiment dataset
2. EDA — sentiment distribution, review length analysis
3. Clean text — remove URLs, mentions, stopwords
4. Tokenize and pad sequences
5. Train LSTM model with Early Stopping
6. Evaluate — accuracy, confusion matrix, training curves

## How to Run
pip install -r requirements.txt
python main.py

## Output
All charts saved automatically to /plots folder:
- sentiment_distribution.png
- review_length.png
- training_curves.png
- confusion_matrix.png