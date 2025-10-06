# Toxic Sentiment Analysis in Real-Time Chats

**Authors:** Victoria Grosu, Chiara Pesce, Francesco Perencin and Tommaso Talamo.  
**Date:** 2025  
**Platform:** Google Colab (PyTorch Implementation)

---

## Project Overview
This project aims to implement a **real-time sentiment toxicity detection model**, capable of identifying and classifying toxic comments in live chat environments.  
The model can be applied to **social media**, **customer support chats**, or any other communication platform where user interaction occurs.

### Objectives
- **Real-time detection:** Analyze and flag messages as they appear.  
- **User safety:** Foster positive and respectful communication environments.  
- **Automatic moderation:** Reduce human moderation workload.  
- **Platform flexibility:** Adaptable to various online communities and platforms.  
- **Toxicity classification:** Detect multiple toxicity types (multi-label classification).

---

## Dataset
The dataset originates from **Wikipedia discussion pages** and is designed for **multi-label classification tasks**.

**Files:**
- `train.csv` – labeled training data.  
- `test.csv` – test data for prediction.  
- `test_labels.csv` – evaluation labels (-1 indicates unused samples).

**Toxicity classes:**
1. `toxic`
2. `severe_toxic`
3. `obscene`
4. `threat`
5. `insult`
6. `identity_hate`

---

## Preprocessing
Each preprocessing function plays a role in cleaning and standardizing the text data:

- **Pre-cleaning:** Lowercasing and removing unnecessary characters.  
- **Hashtag and URL removal:** Excludes irrelevant metadata.  
- **Character filtering:** Removes emojis and special symbols.  
- **Text truncation:** Limits sequence length for computational efficiency.  
- **Stemming & Lemmatization:** Normalizes words to their base form.  
- **Punctuation cleanup:** Removes repeated punctuation marks (`!!!`, `???`, etc.).

---

## Tools and Libraries
- **Pandas / NumPy** – data manipulation and numerical operations  
- **NLTK** – tokenization, stemming, lemmatization  
- **Hugging Face Transformers** – implementation and fine-tuning of `BERT`  
- **Scikit-Learn** – metrics and model evaluation  
- **PyTorch** – model building, training, and optimization

---

## Model Design

### Input Representation
Each sentence is represented as a vector of six binary outputs, one per toxicity label.  
The system performs **multi-label classification** (more than one class can be `1`).

### Processing Pipeline
1. Text input is tokenized, lemmatized, and stemmed.  
2. The processed tokens are embedded using **BERT**.  
3. The BERT embeddings are passed to a **bidirectional RNN** (either LSTM or GRU).  
4. A **Vaswani Attention Layer** produces a context vector.  
5. The context vector is concatenated with the **[CLS] token** embedding from BERT.  
6. The concatenated vector is sent to the **classification layer**, producing six binary outputs.

---

## Model Architectures
1. **BERT + BiLSTM + Attention + [CLS]**  
2. **BERT + BiLSTM + Attention (no [CLS])**  
3. **BERT + BiGRU + Attention + [CLS]**  
4. **BERT + BiGRU + Attention (no [CLS])**

---

## BERT Details
Model: `bert-base-uncased`  
- 12 transformer layers  
- 768 hidden units per layer  
- 12 attention heads  
- ~110M parameters  
- Fine-tuned for toxic comment classification → `bert_sequence_classification_trained`

---

## Recurrent Layers
Both **LSTM** and **GRU** architectures were tested.

**Parameters:**
- Embedding dimension: `768`  
- Hidden dimension: `256`  
- Output dimension: `6` (six toxicity types)  
- Training epochs: `30`

**Loss functions tested:**
- `BCEWithLogitsLoss` – combines Sigmoid + Binary Cross Entropy  
- `MultiLabelSoftMarginLoss` – optimizes multi-label separation margins

---

## Experimental Results

### Performance Summary
| Metric | Best Model (GRU + CLS + BCE) | Worst Model (GRU + MultiLabelSoftMargin) |
|--------|------------------------------|------------------------------------------|
| Accuracy | Highest overall | Lowest overall |
| Precision | Very high | 0.9172 |
| Recall | 0.7168 (+9% improvement) | 0.6236 |
| F1-score | Strong and balanced | 0.7424 |
| Hamming Distance | Lowest (best) | Highest (6189) |
| Execution Speed | Fastest | Slower |
| Best Use Case | Real-time classification | — |

---

### Loss Function Comparison
- **`BCEWithLogitsLoss`** → Slightly higher precision, best for models with `[CLS]`  
- **`MultiLabelSoftMarginLoss`** → Higher recall and F1 for LSTM and GRU+CLS  
- **Efficiency:** Nearly identical runtime; GRU+CLS is the most time-efficient for real-time use.  

---

## Metric Analysis
- Changing the **loss function** affects the number of epochs needed for convergence.  
- **GRU models** train faster than LSTM (3–4s faster per 8000 samples).  
- **GRU+CLS** requires fewer epochs and achieves best recall and accuracy.  
- **BCEWithLogitsLoss** yields slightly better precision.  
- **MultiLabelSoftMarginLoss** provides marginally better recall and F1.

**Conclusion:**  
*The GRU + CLS + BCEWithLogitsLoss model achieves the best overall trade-off between accuracy, recall, speed, and interpretability.*

---

## Discussion and Examples
The model successfully identifies toxic and non-toxic comments across multiple contexts.  
It performs well on:
- **Threat detection** disguised in neutral wording  
- **Subtle hate speech** or **masked insults**  
- **Contextual toxicity** (sarcastic or indirect expressions)

However, limitations remain:
- Words intentionally obfuscated (e.g., letter swaps) may bypass detection.  
- Distinguishing **sarcasm** from **actual offensive content** remains challenging.

---

## Future Improvements
- Implement **data augmentation** for underrepresented toxicity classes.  
- Enhance **sarcasm and irony detection**.  
- Introduce **character-level embeddings** to detect obfuscated toxic words.  
- Deploy model as a **real-time moderation API** for chat systems.

---

## Key Takeaways
- The **GRU + CLS + BCEWithLogitsLoss** architecture is optimal for real-time toxicity analysis.  
- Balances **accuracy, efficiency, and interpretability**.  
- Demonstrates that hybrid architectures combining **transformers and RNNs with attention** can outperform standard BERT fine-tuning in nuanced sentiment detection.

---
