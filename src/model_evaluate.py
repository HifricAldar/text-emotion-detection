import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import *

def predict_batch(model, tokenizer, df, max_length=32):
    texts = df['text'].values
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    
    predictions = model.predict(padded_sequences)
    label_indices = np.argmax(predictions, axis=1)
    probabilities = np.max(predictions, axis=1)
    
    predicted_labels = [CLASS_NAMES[idx] for idx in label_indices]
    
    df['predicted_emotion'] = predicted_labels
    df['prediction_probability'] = probabilities
    
    return df

def evaluate_model(model, tokenizer, test_data, max_length=32):
    #CLASS_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    updated_test_df = predict_batch(model, tokenizer, test_data, max_length)
    updated_test_df['emotion'] = updated_test_df['label'].map(lambda x: CLASS_NAMES[x])
    
    accuracy = np.mean(updated_test_df['predicted_emotion'] == updated_test_df['emotion'])
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(updated_test_df['emotion'], updated_test_df['predicted_emotion']))
    
    return updated_test_df

def plot_confusion_matrix(updated_test_df):
    #CLASS_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(updated_test_df['emotion'], updated_test_df['predicted_emotion'], 
                         labels=CLASS_NAMES)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CONFUSION_MATRIX_IMG)
    plt.close()