import matplotlib.pyplot as plt
from config import *
def train_model(model, train_data, train_labels, validation_data, validation_labels, 
                epochs=10, batch_size=32, verbose=2):
    history = model.fit(
        train_data, train_labels,
        epochs=epochs,
        validation_data=(validation_data, validation_labels),
        batch_size=batch_size,
        verbose=verbose
    )
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_GRAPH_IMG)
    plt.close()