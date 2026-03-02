# 📈 Social Media-Driven Stock Movement Prediction

This is a project for the National Taiwan University (NTU) Big Data Analytics (BDA) course. Our team aimed to predict short-term stock price movements using social sentiment data collected from online forums and financial articles, with a focus on Taiwan Semiconductor Manufacturing Company (TSMC, 2330.TW).

台灣大學大數據分析（Big Data Analytics, BDA）課程專案。我們的團隊目標是利用從網路論壇與財經新聞中蒐集的社群情緒資料，以 NLP 機器學習模型訓練並預測股表漲跌、並進行回測，預測股票的短期價格變化，研究重點聚焦於台積電（TSMC, 股票代碼：2330）。

---

## 📌 Project Overview

This project examined whether stock price movements can be predicted using **financial indicators** and **textual signals** from news articles and online discussions, with **TSMC (2330.TW)** as a case study. The research integrated quarterly financial reports with daily trading data and performed systematic cleaning and preprocessing of large-scale news articles and forum discussions. Textual data was represented using **TF-IDF** and **word-based feature selection**, and stock movement prediction was formulated as a **three-class classification task** (rise, neutral, fall). Multiple machine learning models were trained and evaluated, including Random Forest, SVM, KNN, MLP, and ensemble voting methods, to compare predictive performance. **Rolling backtesting** was conducted to simulate real-world investment settings, demonstrating that TF-IDF with feature selection and Random Forest provides stable and competitive results.

### Research Objectives

1. **Data Integration**: Integrate structured financial data (quarterly reports, daily trading data) with unstructured textual data (news articles, forum discussions)
2. **Feature Engineering**: Develop effective text representation methods using TF-IDF and feature selection techniques
3. **Model Comparison**: Evaluate multiple machine learning algorithms for stock movement prediction
4. **Real-world Validation**: Conduct rolling backtesting to assess model performance in realistic trading scenarios

### Data Collection

We collected over **450,000** textual records between **2022.03–2024.01**, consisting of:
- **Financial news articles**: Professional financial news sources covering TSMC-related events
- **Forum discussions**: User-generated content from platforms like PTT (批踢踢實業坊) and StockFeel (股感)
- **Community replies and user comments**: Interactive discussions and sentiment expressions

Additionally, we integrated:
- **Quarterly financial reports**: TSMC's quarterly financial statements and key metrics
- **Daily trading data**: Historical stock prices, volumes, and market indicators

---

## 🧠 Methodology

### 1. Data Preprocessing

#### Text Data Cleaning
- **Duplicate removal**: Identified and removed duplicate articles and posts
- **Noise removal**: Cleaned emojis, HTML tags, special symbols, and formatting artifacts
- **Content merging**: Aggregated main posts with their replies and comments to capture full discussion context
- **Date alignment**: Mapped textual data to corresponding trading days

#### Financial Data Integration
- **Quarterly report mapping**: Aligned quarterly financial indicators (revenue, profit margins, etc.) with daily stock data
- **Temporal alignment**: Ensured financial metrics were properly forward-filled to match daily trading periods
- **Feature normalization**: Standardized financial indicators for consistent scaling

#### Label Generation
Our goal was to predict whether the stock price would **rise**, **remain flat**, or **fall** within the next **3 trading days**. Labels were generated as follows:
- **Rise (↑)**: Price increase > threshold (e.g., > 2%)
- **Neutral (→)**: Price change within neutral range (e.g., -2% to +2%)
- **Fall (↓)**: Price decrease > threshold (e.g., < -2%)

### 2. Feature Engineering

#### Text Preprocessing
- **Word segmentation**: Used `Monpa` (蒙巴中文斷詞系統) for Traditional Chinese text segmentation
- **Stop word removal**: Filtered common words that don't carry predictive information
- **Text normalization**: Standardized text format and encoding

#### Text Vectorization

**TF-IDF (Term Frequency-Inverse Document Frequency)**
- Converted segmented text into TF-IDF vectors
- Captured term importance relative to document frequency
- Generated high-dimensional sparse feature matrices

**Feature Selection**
- Applied **Chi-square (χ²) test** for feature selection
- Selected top-k most informative features based on statistical significance
- Reduced dimensionality while preserving predictive power
- Improved model interpretability and reduced overfitting risk

**BERT Embeddings (Alternative Approach)**
- Fine-tuned BERT models for financial domain
- Generated dense contextual embeddings
- Compared performance with TF-IDF-based approaches

### 3. Classification Models

We experimented with multiple machine learning algorithms to compare their predictive performance:

- **Random Forest**: Ensemble of decision trees, robust to overfitting, provides feature importance
- **Support Vector Machine (SVM)**: Effective for high-dimensional sparse data (TF-IDF features)
- **K-Nearest Neighbors (KNN)**: Instance-based learning, suitable for local patterns
- **Multi-Layer Perceptron (MLP)**: Deep neural network capable of learning complex non-linear relationships
- **XGBoost**: Gradient boosting framework with regularization
- **Voting Classifier (Ensemble)**: Combined predictions from multiple models using majority voting

#### Model Training Strategy
- **Train-validation-test split**: Temporal split respecting time order
- **Hyperparameter tuning**: Grid search and cross-validation for optimal parameters
- **Class balancing**: Addressed imbalanced class distribution using appropriate techniques
- **Evaluation metrics**: Accuracy, precision, recall, F1-score, and confusion matrices

---

## 🏆 Results

### Model Performance Comparison

#### TF-IDF with Feature Selection

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| **Voting Ensemble** | **82.17%** | - | - | - |
| **MLP**             | **81.85%** | - | - | - |
| **Random Forest**    | 79.12%   | - | - | - |
| **SVM**              | 75.89%   | - | - | - |
| **KNN**              | -        | - | - | - |

**Key Findings:**
- **TF-IDF with feature selection** proved to be the most effective text representation method
- **Ensemble voting** achieved the best overall accuracy by combining multiple models
- **MLP** showed strong performance, indicating non-linear patterns in the data
- **Random Forest** provided stable and interpretable results with feature importance insights

#### BERT-Based Models

We also evaluated BERT-based models for comparison. However, due to:
- **Noisy embeddings**: Financial domain-specific noise in pre-trained embeddings
- **Feature imbalance**: Imbalanced class distribution affecting embedding quality
- **Computational complexity**: Higher training and inference costs

The accuracy was generally lower (~60%), suggesting that TF-IDF with careful feature selection is more suitable for this task.

### Rolling Backtesting

To simulate real-world investment scenarios, we conducted **rolling backtesting** with the following methodology:

#### Backtesting Strategy
- **Rolling window**: Used expanding or fixed-size rolling windows for training
- **Out-of-sample testing**: Tested on future periods not seen during training
- **Temporal order preservation**: Maintained chronological order to avoid look-ahead bias
- **Performance metrics**: Evaluated cumulative returns, Sharpe ratio, and maximum drawdown

#### Key Results
- **TF-IDF + Feature Selection + Random Forest** demonstrated stable and competitive performance in backtesting
- Model predictions showed consistent performance across different market conditions
- Feature selection helped reduce overfitting and improved generalization to unseen data

### Model Interpretability

- **Feature importance**: Random Forest provided insights into which words/features are most predictive
- **Chi-square analysis**: Identified statistically significant textual signals
- **Error analysis**: Examined misclassification patterns to understand model limitations

---

## 🔧 Environment & Setup

### Requirements

- **Python**: 3.8 or higher
- **Key Packages**:
  - `scikit-learn`: Machine learning models and utilities
  - `monpa`: Traditional Chinese word segmentation
  - `transformers`: BERT model implementation
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computations
  - `matplotlib`: Visualization
  - `seaborn`: Statistical visualizations
  - `xgboost`: Gradient boosting framework
  - `jupyter`: Notebook environment

### Installation

To install dependencies:

```bash
pip install -r requirements.txt
```

### Data Requirements

- **Textual data**: News articles and forum discussions (CSV format)
- **Financial data**: Quarterly reports and daily trading data
- **Stock data**: Historical price and volume data for TSMC (2330.TW)

---

## 📂 Project Structure

```
Stock-Movement-NLP-Prediction/
├── data/                    # Data storage
│   ├── raw/                 # Raw collected data
│   ├── processed/           # Cleaned and preprocessed data
│   └── financial/          # Financial reports and trading data
├── models/                  # Saved models and vectorizers
│   ├── trained_models/     # Serialized ML models
│   └── vectorizers/        # TF-IDF vectorizers and feature selectors
├── src/                     # Source code
│   ├── preprocessing.py    # Data cleaning and preprocessing
│   ├── vectorizer.py       # TF-IDF and BERT vectorization
│   ├── feature_selection.py # Chi-square feature selection
│   ├── train_model.py      # Model training pipeline
│   └── backtesting.py      # Rolling backtesting implementation
├── notebooks/               # Jupyter notebooks for analysis
│   ├── 文字前處理(preprocess).ipynb
│   ├── 向量:模型 (BERT).ipynb
│   ├── 向量:模型 (無特徵之TF_IDF).ipynb
│   ├── 向量:模型:回測(特徵關鍵字TF_IDF).ipynb
│   └── 回測(無特徵:之TF-IDF).ipynb
├── results/                 # Results and visualizations
│   ├── metrics/            # Evaluation metrics
│   ├── plots/              # Performance plots and charts
│   └── backtest_results/   # Backtesting results
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

### Usage

1. **Data Preprocessing**:
   ```python
   python src/preprocessing.py
   ```

2. **Feature Engineering**:
   ```python
   python src/vectorizer.py
   ```

3. **Model Training**:
   ```python
   python src/train_model.py
   ```

4. **Backtesting**:
   ```python
   python src/backtesting.py
   ```

---

## 📊 Research Methodology Summary

This project follows a systematic approach to stock movement prediction:

1. **Data Collection & Integration**: Gathered 450,000+ textual records and integrated with financial data
2. **Data Preprocessing**: Systematic cleaning, normalization, and temporal alignment
3. **Feature Engineering**: TF-IDF vectorization with statistical feature selection
4. **Model Development**: Training and evaluation of multiple ML algorithms
5. **Validation**: Rolling backtesting to simulate real-world performance
6. **Analysis**: Comparative evaluation and interpretation of results

The research demonstrates that **traditional NLP techniques (TF-IDF) combined with careful feature selection can outperform more complex deep learning approaches** for this specific task, providing a practical and interpretable solution for stock movement prediction.

---

## 📽 Demo & Report

- [📊 Slide deck](https://reurl.cc/EVbR9v)
- [📹 Presentation video](https://reurl.cc/1Kkm5X)

---

## 📚 References & Related Work

This project builds upon research in:
- **Financial NLP**: Text mining for financial prediction
- **Sentiment Analysis**: Extracting sentiment from social media and news
- **Stock Prediction**: Machine learning approaches to price movement forecasting
- **Feature Selection**: Statistical methods for dimensionality reduction

---

## 👥 Contributors

This project was developed as part of the National Taiwan University (NTU) Big Data Analytics (BDA) course.

---

## 📄 License

This project is for educational purposes as part of the NTU BDA course.

---

## 🔬 Technical Details

### Feature Selection Methodology

**Chi-square (χ²) Test for Feature Selection:**
- Calculated chi-square statistic between each feature and target classes
- Selected top-k features with highest statistical significance
- Reduced feature space from thousands to hundreds of most informative terms
- Improved model training speed and generalization

### Model Training Details

- **Cross-validation**: Time-series cross-validation to respect temporal order
- **Hyperparameter optimization**: Grid search for optimal model parameters
- **Class balancing**: Addressed imbalanced classes using SMOTE or class weights
- **Early stopping**: Prevented overfitting in neural network models

### Evaluation Framework

- **Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Temporal validation**: Ensured no data leakage by maintaining time order
- **Statistical significance**: Conducted significance tests to validate improvements

---

## 🎯 Key Contributions

1. **Systematic Data Integration**: Successfully integrated structured financial data with unstructured textual data
2. **Effective Feature Engineering**: Demonstrated that TF-IDF with feature selection outperforms complex embeddings for this task
3. **Comprehensive Model Comparison**: Evaluated multiple ML algorithms to identify best-performing approaches
4. **Real-world Validation**: Conducted rolling backtesting to validate model performance in realistic scenarios
5. **Practical Insights**: Provided actionable findings for stock movement prediction using NLP techniques

---

## ⚠️ Challenges & Limitations

### Data Challenges
- **Data quality**: Noisy user-generated content from forums
- **Temporal alignment**: Matching textual events with stock price movements
- **Class imbalance**: Uneven distribution of rise/neutral/fall classes

### Model Limitations
- **Market efficiency**: Stock markets are highly efficient, making prediction inherently difficult
- **External factors**: Model may not capture all market-moving events
- **Overfitting risk**: High-dimensional text features require careful regularization

### Domain-Specific Issues
- **Chinese NLP**: Limited pre-trained models for Traditional Chinese financial text
- **Context understanding**: Financial jargon and domain-specific terminology
- **Sentiment ambiguity**: Same text may have different interpretations in different contexts

---

## 💡 Future Improvements

### Short-term Enhancements
- **Sentiment scoring**: Apply sentiment analysis with weights based on source credibility
- **Technical indicators**: Integrate technical analysis indicators (RSI, MACD, etc.) with text-based features
- **BERT optimization**: Tune BERT input with tighter filtering and pre-label validation
- **Feature engineering**: Explore n-grams, topic modeling (LDA), and domain-specific embeddings

### Long-term Research Directions
- **Multi-stock prediction**: Extend model to predict movements for multiple stocks
- **Real-time prediction**: Develop real-time prediction pipeline for live trading
- **Deep learning**: Explore LSTM, GRU, or Transformer architectures for sequential modeling
- **Ensemble methods**: Advanced ensemble techniques (stacking, boosting) for improved performance
- **Explainable AI**: Enhance model interpretability with SHAP values or attention mechanisms
- **Alternative data**: Incorporate social media metrics, search trends, and other alternative data sources
