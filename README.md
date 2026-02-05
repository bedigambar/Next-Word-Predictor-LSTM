# Next Word Predictor LSTM

A deep learning project that uses Long Short-Term Memory (LSTM) neural networks to predict the next word in a sequence. The model is trained on Medium articles from the "Towards Data Science" publication to learn patterns in natural language.

## 🎯 Overview

This project implements a next-word prediction system using PyTorch and LSTM networks. Given a sequence of words, the model predicts the most likely next word(s) based on patterns learned from a dataset of Medium articles.

### Example Output
```
Input:  "a beginners guide to word embedding"
Output: "a beginners guide to word embedding with gensim wordvec model using machine learning"
```

## 🚀 Features

- **LSTM-based Architecture**: 2-layer LSTM network with dropout for regularization
- **Word Embeddings**: 200-dimensional word embeddings
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Live Text Generation**: Real-time word-by-word text generation with visual feedback
- **Top-K Predictions**: Can return multiple prediction candidates

## 📋 Requirements

```python
pandas
torch
re
tqdm
```

## 🏗️ Model Architecture

- **Input**: Sequence of 5 words (SEQ_LENGTH = 5)
- **Embedding Layer**: vocab_size × 200
- **LSTM Layers**: 2 layers with 256 hidden units
- **Dropout**: 0.3 dropout rate for regularization
- **Output Layer**: Fully connected layer mapping to vocabulary size

```
Vocabulary Size: 10,422 unique words
Training Samples: 46,700 sequences
```

## 📊 Dataset

The model is trained on [`medium_data.csv`](https://github.com/bedigambar/Next-Word-Predictor-LSTM/blob/main/medium_data.csv), which contains articles from Medium publications including:
- Article titles and subtitles
- Metadata (claps, responses, reading time)
- Publication dates
- Full article text

The dataset includes 6,508 articles primarily from:
- Towards Data Science
- UX Collective
- Better Marketing
- The Startup
- And other Medium publications

## 🔧 How It Works

### 1. Data Preprocessing
- Combines article titles and subtitles
- Cleans text (lowercase, removes special characters)
- Tokenizes text into words
- Builds vocabulary mapping (word ↔ index)

### 2. Sequence Generation
- Creates sliding window sequences of 5 words
- Each sequence is used to predict the 6th word
- Converts words to numerical indices

### 3. Model Training
- Batch size: 64
- Loss function: Cross-Entropy Loss
- Optimizer: Adam (learning rate: 0.001)
- Training epochs: Up to 70 (with early stopping)
- Early stopping patience: 3 epochs

### 4. Prediction
The model can predict in two modes:
- **Single prediction**: Returns the most likely next word
- **Top-K prediction**: Returns K most probable next words

## 📈 Training Results

The model achieved impressive convergence:
- Initial Loss: 7.2145
- Final Loss: 0.1698 (after 43 epochs)
- Training stopped early due to convergence

## 🎮 Usage

### Training the Model

```python
# Load and preprocess data
df = pd.read_csv("/content/medium_data.csv")
df["text"] = df["title"].fillna("") + " " + df["subtitle"].fillna("")

# Train the model (runs automatically in notebook)
# Training includes early stopping and loss monitoring
```

### Making Predictions

```python
# Predict next word
text = "a beginners guide to word embedding"
predictions = predict_next_word(model, text, top_k=3)
print(predictions)

# Generate extended text
for _ in range(10):
    preds = predict_next_word(model, text, top_k=1)
    next_word = preds[0]
    text += " " + next_word
```

## 🔬 Model Parameters

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 200 |
| Hidden Dimension | 256 |
| Number of Layers | 2 |
| Dropout Rate | 0.3 |
| Sequence Length | 5 |
| Batch Size | 64 |
| Learning Rate | 0.001 |

## 🎓 Technical Details

### NextWordLSTM Class
```python
class NextWordLSTM(nn.Module):
    - Embedding layer for word representations
    - 2-layer LSTM with batch_first=True
    - Fully connected output layer
    - Forward pass returns logits for vocabulary
```

### Prediction Function
- Cleans and tokenizes input text
- Takes last SEQ_LENGTH words
- Returns top-K predicted words with softmax probabilities

## 📝 File Structure

```
Next-Word-Predictor-LSTM/
├── NextWordPredictor.ipynb    # Main notebook with complete implementation
└── medium_data.csv             # Training dataset (Medium articles)
```

## 🎯 Future Improvements

- [ ] Implement beam search for better text generation
- [ ] Add temperature parameter for controlling randomness
- [ ] Experiment with different sequence lengths
- [ ] Try transformer-based architectures
- [ ] Add model checkpointing
- [ ] Implement validation set evaluation
- [ ] Add more diverse datasets

## 📜 License

This project is open source and available for educational purposes [LICENSE](https://github.com/bedigambar/Next-Word-Predictor-LSTM/blob/main/LICENSE)

## 🙏 Acknowledgments

- Dataset sourced from [Medium articles](https://github.com/bedigambar/Next-Word-Predictor-LSTM/blob/main/medium_data.csv)
- Built with PyTorch deep learning framework

---

**Note**: This model is trained on Medium articles and works best with data science, technology, and business-related content.
