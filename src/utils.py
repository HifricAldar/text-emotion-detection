import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import *

def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def predict_emotion(text, model, tokenizer, max_length=32):
    #CLASS_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length)
    
    prediction = model.predict(padded)
    label_index = np.argmax(prediction[0])
    probability = np.max(prediction[0])
    
    return CLASS_NAMES[label_index], probability