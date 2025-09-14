import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from PIL import Image


with open("./results/models/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("results/models/emotion_classification_model.h5")

class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

icon_paths = {
    "sadness": "image emoji/sadness.png",
    "joy": "image emoji/joy.png",
    "love": "image emoji/love.png",
    "anger": "image emoji/anger.png",
    "fear": "image emoji/fear.png",
    "surprise": "image emoji/surprise.png",
}

def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=32, padding="post", truncating="post")
    pred = model.predict(padded)[0]
    idx = np.argmax(pred)
    emotion = class_names[idx]
    print(f"Text: {text}")
    print(f"Predicted Emotion: {emotion} ({pred[idx]*100:.2f}%)")

    img = Image.open(icon_paths[emotion])
    plt.imshow(img)
    plt.axis("off")
    plt.title(emotion)
    plt.draw()
    plt.pause(5)
    plt.close()     

predict_emotion("I just got accepted into my dream university!")
predict_emotion("I feel really lonely and sad.")
