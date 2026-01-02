"""
Inference script for Transformer-based sentiment classifier.

This script loads the trained model and config, then predicts sentiment
for input text(s). It uses the tuned decision threshold from config.json
to convert probabilities into binary labels.
"""

import json
import re
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os

# Configuration
MODEL_PATH = 'model_checkpoints/best_model.keras'
CONFIG_PATH = 'model_checkpoints/config.json'
TEXT_VECTORIZER_PATH = 'model_checkpoints/text_vectorizer.keras'
MAX_TOKENS = 10000
OUTPUT_SEQUENCE_LENGTH = 341  # Sequence length used during training


def preprocess_text(text):
    """
    Clean and preprocess text for sentiment analysis.
    This function should match the preprocessing used during training.
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text string
    """
    # Convert to lowercase - ensures consistency in text analysis
    text = str(text).lower()
    
    # Remove URLs - URLs don't contribute to sentiment analysis
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove HTML tags if any - cleans any HTML entities from text
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits, keep only letters and spaces
    # This helps focus on actual words for sentiment analysis
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace - replaces multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def load_model_and_config():
    """Load the trained model and configuration."""
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Load config to get decision threshold
    print(f"\nLoading configuration from {CONFIG_PATH}...")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    threshold = config['decision_threshold']
    print(f"Decision threshold: {threshold}")
    print(f"  {config['threshold_description']}")
    
    # Load TextVectorization layer (saved as a model wrapper)
    print(f"\nLoading TextVectorization layer from {TEXT_VECTORIZER_PATH}...")
    if not os.path.exists(TEXT_VECTORIZER_PATH):
        raise FileNotFoundError(
            f"TextVectorization layer not found at {TEXT_VECTORIZER_PATH}.\n"
            "Please run the notebook cell that saves the TextVectorization layer."
        )
    # Load the wrapper model (which contains the TextVectorization layer)
    vectorizer_model = tf.keras.models.load_model(TEXT_VECTORIZER_PATH)
    # Use the wrapper model directly (it will handle the TextVectorization internally)
    text_vectorizer = vectorizer_model
    
    print("TextVectorization layer loaded successfully!")
    
    return model, text_vectorizer, threshold


def predict_sentiment(text, model, text_vectorizer, threshold):
    """
    Predict sentiment for a single text input.
    
    Args:
        text: Input text string
        model: Loaded Keras model
        text_vectorizer: Loaded TextVectorization layer (or model wrapper)
        threshold: Decision threshold for binary classification
        
    Returns:
        tuple: (probability, binary_label, sentiment_label)
    """
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Vectorize text using the TextVectorization wrapper model
    # The text_vectorizer is a model wrapper containing the TextVectorization layer
    text_tensor = tf.constant([cleaned_text], dtype=tf.string)  # Shape: [1]
    vectorized = text_vectorizer(text_tensor)  # Shape: [1, sequence_length]
    
    # Make prediction
    probability = model.predict(vectorized, verbose=0)[0][0]
    
    # Convert probability to binary label using threshold
    binary_label = 1 if probability > threshold else 0
    sentiment_label = "Positive" if binary_label == 1 else "Negative"
    
    return probability, binary_label, sentiment_label


def main():
    """Main inference function."""
    # Load model, vectorizer, and config
    model, text_vectorizer, threshold = load_model_and_config()
    
    # Example inputs (can be replaced with command-line arguments or file input)
    print("\n" + "=" * 70)
    print("SENTIMENT PREDICTION")
    print("=" * 70)
    
    # Example texts
    example_texts = [
        "This product is amazing! I love it so much. Highly recommended!",
        "Terrible quality. Waste of money. Very disappointed.",
        "It's okay, nothing special but it works fine.",
        "I absolutely hate this. Worst purchase ever.",
        "Great value for money. Very satisfied with my purchase."
    ]
    
    print(f"\nProcessing {len(example_texts)} example texts...\n")
    
    # Predict sentiment for each text
    for i, text in enumerate(example_texts, 1):
        probability, binary_label, sentiment_label = predict_sentiment(
            text, model, text_vectorizer, threshold
        )
        
        print(f"Text {i}:")
        print(f"  Input: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  Probability: {probability:.4f} ({probability*100:.2f}%)")
        print(f"  Binary Label: {binary_label}")
        print(f"  Sentiment: {sentiment_label}")
        print()
    
    print("=" * 70)

if __name__ == "__main__":
    main()

