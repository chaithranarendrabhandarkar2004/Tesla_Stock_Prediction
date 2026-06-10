# 📈 Tesla (TSLA) Stock Price Prediction & Analytics Dashboard

A state-of-the-art time-series analysis and forecasting dashboard powered by Deep Learning (**Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** networks). The application features a premium dark-themed Streamlit UI, interactive Plotly visualizations, autoregressive forecasting, and custom dataset capabilities.

> [!NOTE]
> All models (RNN and LSTM) are pre-trained and saved under the `models/` directory. You can run the dashboard immediately or trigger training to overwrite the existing weights.

---

## 🌟 Key Features

*   **Interactive Control Center:** Modify lookback windows (30–90 days) and forecast horizons (5–90 days) dynamically.
*   **Dual Model Support & Comparison:** Evaluate both a standard **Simple RNN** and a stacked **LSTM (Recommended)** network. Compare metrics side by side.
*   **Autoregressive Forecasting:** Predict future trading days iteratively by feeding predictions back into the model sequences.
*   **Upload Custom Datasets:** Drag and drop any custom stock CSV file or use the built-in Tesla (TSLA) dataset.
*   **Glassmorphic KPI Metrics:** Clear, real-time performance evaluation using **RMSE, MAE, MSE, and $R^2$ Score** cards.
*   **Interactive Graphics:** Smooth, responsive Plotly charts representing historical trends, test-set prediction fits, trading volume distributions, and forecasted horizons.
*   **Architecture Educational Panel:** Code snippets and architectural summaries detailing time-series sequence training.

---

## 📂 Project Directory Structure

```text
Tesla_Stock_Prediction/
├── .venv/                     # Python Virtual Environment
├── components/                # Custom React components (Aurora background framework)
├── data/                      # Historical datasets
│   └── TSLA.csv               # Default Tesla historical dataset
├── models/                    # Serialized models & parameters
│   ├── best_lstm.keras        # Trained LSTM Model
│   ├── rnn_model.keras        # Trained Simple RNN Model
│   └── scaler.pkl             # Pre-fit MinMaxScaler
├── src/                       # Source modules
│   ├── data_preprocessing.py  # Loading & cleaning CSV data
│   ├── feature_engineering.py # Window sequencing helpers
│   ├── train_rnn.py           # RNN architecture compile
│   ├── train_lstm.py          # LSTM architecture compile
│   ├── predict.py             # Evaluation metric calculations
│   └── utils.py               # Miscellaneous utility helpers
├── app.py                     # Main Streamlit web application
├── main.py                    # Training orchestration script
├── pyproject.toml             # Python project definition
├── requirements.txt           # Standard pip package requirements
└── README.md                  # Project Documentation (This File)
```

---

## 🧠 Model Architectures

### 1. Stacked Long Short-Term Memory (LSTM)
Designed to combat the vanishing gradient problem, the LSTM utilizes memory cell states and gates to capture long-term dependencies:
*   **Layer 1:** LSTM (64 units, returning sequences)
*   **Regularization:** Dropout (0.20)
*   **Layer 2:** LSTM (64 units)
*   **Regularization:** Dropout (0.20)
*   **Output Layer:** Dense (1 unit, linear activation)

### 2. Simple Recurrent Neural Network (RNN)
An agile network suitable for shorter-term sequences:
*   **Layer 1:** SimpleRNN (50 units)
*   **Regularization:** Dropout (0.20)
*   **Output Layer:** Dense (1 unit, linear activation)

---

## 🚀 How to Run the Project

### Option A: Using the Virtual Environment (Recommended)

1. **Activate the virtual environment:**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
2. **Launch the dashboard:**
   ```powershell
   streamlit run app.py
   ```

### Option B: Using the `uv` Tool

If you prefer using `uv` to orchestrate runs:
```powershell
uv run streamlit run app.py
```

### Model Re-training (Optional)

To retrain the models with fresh parameters or update the scaler:
```powershell
# Standard:
python main.py

# Using uv:
uv run python main.py
```

---

## 📊 Analytics Metrics Explained

*   **RMSE (Root Mean Squared Error):** Represents the standard deviation of residuals (prediction errors). Lower values indicate smaller errors in predicting dollar values.
*   **MAE (Mean Absolute Error):** The average absolute difference between the actual and predicted price.
*   **$R^2$ Score (Coefficient of Determination):** Explains how well the model predicts stock price variances. A score closer to `1.0` indicates a highly accurate regression fit.
