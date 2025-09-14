# 🧠 Emotion Classification (Single-Label)
This project is about building a **text-based emotion classification** model using deep learning. The model predicts **one of six emotions** from short text inputs such as messages, tweets, or short phrases.

| Emotion   | Example Text                       |
|-----------|------------------------------------|
| 😢 Sadness | "I felt lonely after coming home." |
| 😀 Joy     | "I was so happy to see my friend." |
| ❤️ Love    | "Spending time with family felt warm." |
| 😡 Anger   | "I was frustrated with the delay." |
| 😨 Fear    | "I was scared walking alone at night." |
| 😲 Surprise| "I didn’t expect that news at all." |

# 📊 Dataset
- Source: CARER Dataset [Saravia et al., 2018](https://www.aclweb.org/anthology/D18-1404)
- Original size: **~416,809** samples
- This project uses a balanced subset with **15,000** samples per class.
     - The smallest emotion class in the dataset contains around 15k samples.

     - To ensure balance, each class is capped at 15k samples.

     - Final dataset size: **90,000 samples (6 classes × 15k)**  

If you use this dataset, please cite the authors:

```bibtex
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697"
}
```
Dataset is also available on [Hugging Face Datasets](https://huggingface.co/datasets/dair-ai/emotion)


## ⚙️ Preprocessing

1. **Drop duplicates**  
2. **Balance dataset** → 15k per class  
3. **Tokenization** with OOV handling (`<OOV>`)  
4. **Padding** to max length = 32  
5. **One-hot encoding** for labels  


## 🧠 Model Architecture

The model is built with **Keras (TensorFlow backend)**:

- Embedding layer  
- Bidirectional LSTM  
- Dense layers with ReLU activation  
- Softmax output for classification  

**Training setup:**
- Optimizer: `Adam`  
- Loss: `CategoricalCrossentropy`  
- Metrics: `Accuracy`  
- Epochs: Configurable (default = 10)  
- Batch size: 32  


## 📈 Results

- **Training & Validation Accuracy/Loss**

![Accuracy & Loss](results\graphics\training_history.png)

- **Confusion Matrix**

![Confusion Matrix](results\graphics\confusion_matrix.png)
- **Saved Models**: `.h5` (Keras) and `.pkl` (Tokenizer & preprocessing pipeline)  

⚠️ Note: Heavy artifacts (`models/`.etc) are ignored in Git for repository cleanliness.  

---

## 🚧 Limitations

- Dataset only contains **short text snippets** (1–2 sentences).  
- Model performance may degrade on **long-form text** (e.g., journals, diaries).  
- Only **single-label classification** → cannot capture multiple emotions in a single text.  

---

## 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/HifricAldar/text-emotion-detection.git

pip install -r requirements.txt