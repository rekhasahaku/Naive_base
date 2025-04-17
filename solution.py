import re
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB


def preprocess_review(text):
    """Lowercases and splits review into words."""
    return re.split(r'[ ,.\n:;\|/]', text.lower())


def build_feature_vector(bag_of_words, review):
    """Counts keyword occurrences in a review."""
    words = preprocess_review(review)
    return [words.count(word) for word in bag_of_words]


def prepare_feature_matrix(data, column, bag_of_words):
    """Constructs a feature matrix for a given dataset."""
    matrix = []
    for review in data[column]:
        vector = build_feature_vector(bag_of_words, review)
        print(vector)  # Optional: print for inspection
        matrix.append(vector)
    return matrix


def print_prediction_results(predictions, probabilities):
    """Prints class prediction and probabilities for each test sample."""
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"\nTEST #{i + 1}")
        print(f"Probabilities: {prob[0]:.4f} (Negative) vs {prob[1]:.4f} (Positive)")
        label = 'Positive' if pred == 1 else 'Negative'
        print(f"Prediction: {pred} ({label})")


def manual_probability_debug():
    """Manual debug calculation of probabilities for learning purposes."""
    print("\nManual Posterior Probabilities (Debug):")

    # Test 1
    p1_pos = 6 / 10 * 4 / 15 * 1 * 3 / 15 * 1 / 15
    p1_neg = 4 / 10 * 2 / 12 * 1 * 5 / 12 * 4 / 12
    norm_1 = p1_pos + p1_neg
    print("Test Review #1:")
    print(f"  P(Negative): {p1_neg / norm_1:.4f}, P(Positive): {p1_pos / norm_1:.4f}")

    # Test 2
    p2_pos = 6 / 10 * 4 / 15 * 7 / 15 * 3 / 15 * 1 / 15
    p2_neg = 4 / 10 * 2 / 12 * 1 / 12 * 5 / 12 * 4 / 12
    norm_2 = p2_pos + p2_neg
    print("Test Review #2:")
    print(f"  P(Negative): {p2_neg / norm_2:.4f}, P(Positive): {p2_pos / norm_2:.4f}")


def main():
    # Define the 4 keywords for bag-of-words feature extraction
    keywords = ['great', 'happy', 'bad', 'return']

    # Load datasets
    train_df = pd.read_csv("training_reviews.csv", encoding='cp1252')
    test_df = pd.read_csv("test_reviews.csv", encoding='cp1252')

    # Feature matrices
    X_train = prepare_feature_matrix(train_df, 'reviews', keywords)
    X_test = prepare_feature_matrix(test_df, 'test_reviews', keywords)

    # Labels
    y_train = train_df['class'].tolist()
    print("\nTraining Labels:", y_train)

    # Train Naive Bayes classifier
    model = MultinomialNB(alpha=1e-10, fit_prior=True)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\nPredicted Classes:", y_pred)
    print("Predicted Probabilities:\n", y_proba)

    # Print test results
    print_prediction_results(y_pred, y_proba)

    # Optional: Manual probability trace
    manual_probability_debug()


if __name__ == "__main__":
    main()
