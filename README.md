# Tesla Stock Analytics & Prediction Dashboard

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-red)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/library-TensorFlow%20%2F%20Keras-orange)](https://tensorflow.org/)

A state-of-the-art visual analytics dashboard powered by Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks to analyze and forecast Tesla (TSLA) stock prices.

---

## 🌟 Key Features

- **Interactive Control Center:** Configure lookback periods (30–90 days) and forecasting horizons (5–90 days) in real time.
- **Deep Learning Predictions:** Compare predictions from a **Simple RNN** and a **Long Short-Term Memory (LSTM)** network.
- **Autoregressive Forecasting:** Project future TSLA stock prices for custom horizons dynamically.
- **Statistical Analytics & KPIs:** Track model metrics like Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ Score on interactive cards.
- **Dataset Explorer:** Browse historical data summaries, raw datasets, and analyze trading volumes.
- **Premium Glassmorphic UI:** Features custom dark mode CSS, subtle glowing animations, and responsive Plotly visual charts.

---

## 📂 Project Directory Structure

```text
Tesla_Stock_Prediction/
├── .venv/                   # Virtual environment (ignored by Git)
├── components/              # Modern UI components
│   └── ui/
│       ├── aurora-background.tsx
│       └── demo.tsx
├── data/
│   └── TSLA.csv             # Historical Tesla stock dataset
├── lib/
│   └── utils.ts             # TS helper functions
├── models/                  # Saved pre-trained models
│   ├── best_lstm.keras      # Pre-trained LSTM weights
│   ├── rnn_model.keras      # Pre-trained Simple RNN weights
│   └── scaler.pkl           # MinMaxScaler state
├── notebooks/
│   └── analysis.ipynb       # Jupyter notebook for exploratory data analysis
├── src/                     # Core python module code
│   ├── data_preprocessing.py # Load, clean, and scale stock data
│   ├── feature_engineering.py # Create temporal sliding window sequences
│   ├── train_lstm.py        # LSTM neural network configuration
│   ├── train_rnn.py         # Simple RNN neural network configuration
│   ├── predict.py           # Evaluation helper (RMSE, MSE, R2)
│   └── utils.ts
├── app.py                   # Main Streamlit dashboard application
├── main.py                  # Training pipeline coordinator script
├── requirements.txt         # Core dependencies
└── pyproject.toml           # Package configuration metadata
```

---

## 🧠 Model Architectures

### 1. Simple Recurrent Neural Network (RNN)
Designed for lightweight and fast sequence modeling.
- **Structure:**
  - `SimpleRNN` layer (50 units, input window of 60 trading days)
  - `Dropout` layer (rate = 0.2) to prevent overfitting
  - `Dense` layer outputting 1 predicted stock price

### 2. Long Short-Term Memory (LSTM)
Designed to combat the vanishing gradient problem and capture long-term trends.
- **Structure:**
  - `LSTM` layer (64 units, returning sequences)
  - `Dropout` layer (rate = 0.2)
  - `LSTM` layer (64 units)
  - `Dropout` layer (rate = 0.2)
  - `Dense` layer outputting 1 predicted stock price

---

## 🚀 Getting Started

### Prerequisites
- Python `>= 3.12` installed.

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chaithranarendrabhandarkar2004/Tesla_Stock_Prediction.git
   cd Tesla_Stock_Prediction
   ```

2. **Create and Activate Virtual Environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   * **macOS / Linux:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Run the Project

### 1. Start the Dashboard UI
Launch the interactive visual dashboard on localhost:
```bash
streamlit run app.py
```

### 2. Retrain Models
To run the full training pipeline and save updated models/scalers to the `models/` directory:
```bash
python main.py
```

---

## ⚡ Optional: Run with `uv`
If you have [uv](https://github.com/astral-sh/uv) installed, you can skip virtual environment management:

- **Launch Dashboard:**
  ```bash
  uv run streamlit run app.py
  ```
- **Train Models:**
  ```bash
  uv run python main.py
  ```
