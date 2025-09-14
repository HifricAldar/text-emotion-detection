import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from config import *

def add_emotion_labels(data):
    data['emotion'] = data['label'].map(lambda x: CLASS_NAMES[x])
    return data

def balance_dataset(data, samples_per_class=15000):
    balanced_dfs = []
    
    for emotion in CLASS_NAMES:
        emotion_data = data[data['emotion'] == emotion]
        
        if emotion == 'surprise':
            balanced_dfs.append(emotion_data)
        else:
            if len(emotion_data) > samples_per_class:
                emotion_data = emotion_data.sample(n=samples_per_class, random_state=42)
            balanced_dfs.append(emotion_data)
    
    combined_df = pd.concat(balanced_dfs, ignore_index=True)
    return combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

def split_data(data, test_size=0.2, val_size=0.5):
    text = data['text']
    label = data['label']
    
    data_train_text, data_temp_text, data_train_label, data_temp_label = train_test_split(
        text, label, test_size=test_size, random_state=42
    )
    
    data_test_text, data_val_text, data_test_label, data_val_label = train_test_split(
        data_temp_text, data_temp_label, test_size=val_size, random_state=42
    )
    
    train_data = pd.DataFrame({'text': data_train_text, 'label': data_train_label})
    val_data = pd.DataFrame({'text': data_val_text, 'label': data_val_label})
    test_data = pd.DataFrame({'text': data_test_text, 'label': data_test_label})
    
    return train_data, val_data, test_data
def save_split_datasets(train_data, val_data, test_data):
    train_data.to_csv(TRAIN_PROCESSED_PATH, index=False)
    val_data.to_csv(VAL_PROCESSED_PATH, index=False)
    test_data.to_csv(TEST_PROCESSED_PATH, index=False)


def preprocess_text_data(train_data, val_data, vocab_size=14500, max_length=32):
    oov_tok = "<OOV>"
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_data['text'].values)
    
    training_sequences = tokenizer.texts_to_sequences(train_data['text'].values)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding='post', truncating='post')
    
    validation_sequences = tokenizer.texts_to_sequences(val_data['text'].values)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding='post', truncating='post')
    
    num_classes = len(train_data['label'].unique())
    train_labels = to_categorical(train_data['label'].values, num_classes=num_classes)
    val_labels = to_categorical(val_data['label'].values, num_classes=num_classes)
    
    return tokenizer, training_padded, validation_padded, train_labels, val_labels, num_classes