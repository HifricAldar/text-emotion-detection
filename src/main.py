import os
import sys

# Import semua fungsi dari modul yang diperlukan
from data_loader import load_data, remove_duplicates
from data_preprocessing import add_emotion_labels, balance_dataset, split_data, preprocess_text_data
from model_builder import build_sequential_model, compile_model
from model_training import train_model, plot_training_history
from model_evaluate import evaluate_model, plot_confusion_matrix
from utils import save_tokenizer
from config import *

def main():
    print("Starting Emotion Analysis Pipeline...")
    
    # 1. Load data
    print("Loading data...")
    data = load_data(DATA_PATH)
    data = remove_duplicates(data)
    
    # 2. Preprocess data
    print("Preprocessing data...")
    data = add_emotion_labels(data)
    balanced_data = balance_dataset(data, SAMPLES_PER_CLASS)
    train_data, val_data, test_data = split_data(balanced_data, TEST_SIZE, VAL_SIZE)
    
    # 3. Preprocess text data
    print("Preprocessing text data...")
    tokenizer, training_padded, validation_padded, train_labels, val_labels, num_classes = preprocess_text_data(
        train_data, val_data, VOCAB_SIZE, MAX_LENGTH
    )
    
    # 4. Build and compile model
    print("Building model...")
    model = build_sequential_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH, num_classes)
    model = compile_model(model)
    model.summary()
    
    # 5. Train model
    print("Training model...")
    history = train_model(model, training_padded, train_labels, validation_padded, val_labels, 
                         EPOCHS, BATCH_SIZE)
    plot_training_history(history)
    
    # 6. Evaluate model
    print("Evaluating model...")
    updated_test_df = evaluate_model(model, tokenizer, test_data, MAX_LENGTH)
    plot_confusion_matrix(updated_test_df)
    
    # 7. Save model and tokenizer
    print("Saving model and tokenizer...")
    model.save(MODEL_SAVE_PATH)
    save_tokenizer(tokenizer, TOKENIZER_SAVE_PATH)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()