# 📈 Social Media-Driven Stock Movement Prediction

This is a project for the NTU Big Data Analytics (BDA) course. Our team aimed to predict short-term stock price movements using social sentiment data collected from online forums and financial articles, with a focus on Taiwan Semiconductor Manufacturing Company (TSMC, 2330.TW).

這是一個為了台灣大學大數據分析（Big Data Analytics, BDA）課程所進行的專案。我們的團隊目標是利用從網路論壇與財經新聞中蒐集的社群情緒資料，預測股票的短期價格變化，研究重點聚焦於台積電（TSMC, 股票代碼：2330.TW）。
---

## 📌 Project Overview

We collected over **450,000** textual records between **2022.03–2024.01**, consisting of:
- Financial news articles
- Forum discussions (e.g., PTT, StockFeel)
- Community replies and user comments

Our goal was to use **Natural Language Processing (NLP)** techniques to classify whether the stock price would **rise**, **remain flat**, or **fall** within the next 3 trading days.

---

## 🧠 Methodology

### 1. Data Preprocessing
- Removed duplicates, emojis, HTML tags, and noisy symbols
- Merged main content with replies
- Labeled data based on 3-day future price movement (↑ / → / ↓)
- Mapped financial report data (quarterly) into daily stock data

### 2. Feature Engineering
- Word segmentation using `Monpa`
- Vectorization via:
  - **TF-IDF** + Chi-square filtering
  - **BERT Embeddings** (fine-tuned)

### 3. Classification Models
We experimented with the following ML algorithms:
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- Voting Classifier (Ensemble)

---

## 🏆 Results

| Model               | Accuracy (TF-IDF with feature selection) |
|--------------------|-------------------------------------------|
| MLP                | **81.85%**                                |
| SVM                | 75.89%                                    |
| Random Forest      | 79.12%                                    |
| Voting Ensemble    | **82.17% (Best)**                         |

We also evaluated BERT-based models. However, due to noisy embeddings and feature imbalance, the accuracy was generally lower (~60%).

---

## 🔧 Environment

- Python 3.8+
- Packages: `sklearn`, `monpa`, `transformers`, `pandas`, `matplotlib`, `xgboost`

To install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Folder Structure (Recommended)

```
├── data/               # Raw and processed data
├── models/             # Saved models and vectorizers
├── src/                # Preprocessing, feature engineering, training
│   ├── preprocessing.py
│   ├── vectorizer.py
│   ├── train_model.py
├── notebooks/          # Jupyter notebooks for EDA and experiments
├── results/            # Evaluation metrics and plots
├── README.md
```

---

## 📽 Demo & Report

- [📊 Slide deck](https://drive.google.com/...)
- [📹 Presentation video](https://drive.google.com/...)

---

## 👥 Team

張鈞傑、張鈞翔、黃于軒、楊好涵、程怡慈、林雨潔  
NTU Big Data Analytics | April 2024

---

## 💡 Future Improvements

- Apply sentiment scoring weights based on source credibility
- Integrate technical indicators with text-based features
- Tune BERT input with tighter filtering and pre-label validation
