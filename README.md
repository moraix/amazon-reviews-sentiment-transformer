# Amazon Reviews Sentiment Analysis (Lightweight Transformer)

This project implements an end-to-end sentiment analysis pipeline on Amazon product reviews using a lightweight Transformer encoder built from scratch (no BERT fine-tuning).

## Key Features

- Binary sentiment classification (positive / negative)

- Clean text preprocessing and tokenization with TextVectorization

- Custom Transformer encoder with self-attention

- Severe class imbalance handled via class weighting

- Decision threshold tuned on validation set for better macro-F1

- Reproducible inference with saved model artifacts and threshold

## Dataset

- Amazon product reviews

- Review title + text combined into a single input

- Ratings mapped as:
Rating ≥ 4 → Positive
Rating ≤ 2 → Negative
Rating = 3 removed

## Evaluation

- Focus on F1-score and macro-F1 due to class imbalance

- Validation and test sets kept strictly unseen during training

## Notes

This repository represents an initial complete version of the project.
Further code cleanup, documentation, and experiments will be added in future updates.
