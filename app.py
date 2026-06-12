import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Tesla Stock Analytics & Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"], .stApp {
        background: #080710;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        inset: -50%;
        width: 200%;
        height: 200%;
        background-image: 
            repeating-linear-gradient(100deg, #000000 0%, #000000 7%, transparent 10%, transparent 12%, #000000 16%),
            repeating-linear-gradient(100deg, #3b82f6 10%, #818cf8 15%, #93c5fd 20%, #ddd6fe 25%, #60a5fa 30%);
        background-size: 300% 200%;
        background-position: 50% 50%;
        filter: blur(80px);
        opacity: 0.12;
        animation: aurora 45s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes aurora {
        0% {
            background-position: 50% 50%, 50% 50%;
        }
        100% {
            background-position: 350% 50%, 350% 50%;
        }
    }

    [data-testid="stHeader"], [data-testid="stSidebar"], .main .block-container {
        position: relative;
        z-index: 1;
    }
    
    .kpi-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: #3b82f6;
    }
    .kpi-title {
        font-size: 14px;
        font-weight: 500;
        color: #8b949e;
        text-transform: uppercase;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .kpi-desc {
        font-size: 11px;
        color: #58a6ff;
    }
   
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 30px;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 25px;
    }
    
    .model-badge {
        background-color: #1e293b;
        color: #3b82f6;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #3b82f6;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sequences(data, time_step=60):
    X = []
    y = []
    for i in range(len(data) - time_step):
        X.append(data[i : i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

@st.cache_data
def load_and_preprocess_data(file_path_or_uploaded):
    df = pd.read_csv(file_path_or_uploaded)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.ffill()
    return df

st.markdown("""
    <div class="main-header">
        <span class="model-badge">Deep Learning Engine v1.0</span>
        <h1 style="margin: 0; color: #ffffff; font-size: 32px; font-weight: 700;">Tesla (TSLA) Stock Price Prediction</h1>
        <p style="margin: 10px 0 0 0; color: #8b949e; font-size: 16px;">
            A state-of-the-art visual analytics dashboard powered by Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #ffffff; font-size: 20px; font-weight: 600;">Control Center</h2>
        <p style="color: #8b949e; font-size: 12px;">Configure models and forecasting horizons</p>
    </div>
    <hr style="border-color: rgba(255,255,255,0.1); margin-top: 0;"/>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload Stock CSV Data",
    type=["csv"],
    help="Default Tesla stock data is loaded automatically if no file is uploaded."
)

default_data_path = "data/TSLA.csv"
df = None

if uploaded_file is not None:
    try:
        df = load_and_preprocess_data(uploaded_file)
        st.sidebar.success("Custom dataset uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading custom dataset: {e}")
        if os.path.exists(default_data_path):
            df = load_and_preprocess_data(default_data_path)
            st.sidebar.warning("Fell back to default Tesla dataset.")
else:
    if os.path.exists(default_data_path):
        df = load_and_preprocess_data(default_data_path)
        st.sidebar.info("Using default Tesla (TSLA) historical dataset.")
    else:
        st.error("No dataset found! Please upload a valid CSV file.")

if df is not None:
    st.sidebar.subheader("Neural Network Configurations")
    selected_model_name = st.sidebar.selectbox(
        "Select Prediction Model",
        options=["LSTM (Recommended)", "Simple RNN", "Compare Models"]
    )

    time_step = st.sidebar.slider(
        "Lookback Period (Days)",
        min_value=30,
        max_value=90,
        value=60,
        help="Number of historical days the models look back to predict the next day's price."
    )

    st.sidebar.subheader("Future Forecasting Horizon")
    forecast_days = st.sidebar.slider(
        "Forecast Horizon (Days)",
        min_value=5,
        max_value=90,
        value=30,
        help="Number of future trading days to predict iteratively."
    )

    models_dir = "models"
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    rnn_path = os.path.join(models_dir, "rnn_model.keras")
    lstm_path = os.path.join(models_dir, "best_lstm.keras")

    scaler = None
    rnn_model = None
    lstm_model = None

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        # Fit a new scaler if default not found
        scaler = MinMaxScaler()
        scaler.fit(df[["Close"]])

    try:
        if os.path.exists(rnn_path):
            rnn_model = tf.keras.models.load_model(rnn_path)
        if os.path.exists(lstm_path):
            lstm_model = tf.keras.models.load_model(lstm_path)
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure the models were trained correctly.")

    scaled_data = scaler.transform(df[["Close"]])
    X, y = create_sequences(scaled_data, time_step=time_step)
    
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    dates_all = df.index[time_step:]
    train_len = len(y) - len(y_test)
    test_dates = dates_all[train_len:]
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analytics & Predictions", 
        "🔮 Future Forecasting", 
        "📈 Dataset Explorer",
        "🧠 Neural Network Architectures"
    ])

    with tab1:
        st.subheader("Test Set Performance & Model Comparison")
        predictions = {}
        metrics = {}
        
        if selected_model_name in ["Simple RNN", "Compare Models"] and rnn_model is not None:
            rnn_pred_scaled = rnn_model.predict(X_test, verbose=0)
            rnn_pred = scaler.inverse_transform(rnn_pred_scaled)
            predictions["Simple RNN"] = rnn_pred
        
            y_test_inv = scaler.inverse_transform(y_test)
            mse = np.mean((y_test_inv - rnn_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test_inv - rnn_pred))
            ss_res = np.sum((y_test_inv - rnn_pred) ** 2)
            ss_tot = np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            metrics["Simple RNN"] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

        if selected_model_name in ["LSTM (Recommended)", "Compare Models"] and lstm_model is not None:
            lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
            lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
            predictions["LSTM"] = lstm_pred
            
            y_test_inv = scaler.inverse_transform(y_test)
            mse = np.mean((y_test_inv - lstm_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test_inv - lstm_pred))
            ss_res = np.sum((y_test_inv - lstm_pred) ** 2)
            ss_tot = np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            metrics["LSTM"] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

        y_test_inv = scaler.inverse_transform(y_test)

        if selected_model_name != "Compare Models":
            model_key = "LSTM" if "LSTM" in selected_model_name else "Simple RNN"
            if model_key in metrics:
                m = metrics[model_key]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-title">Root Mean Squared Error (RMSE)</div>
                            <div class="kpi-value">${m['RMSE']:.2f}</div>
                            <div class="kpi-desc">Lower is better</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-title">Mean Absolute Error (MAE)</div>
                            <div class="kpi-value">${m['MAE']:.2f}</div>
                            <div class="kpi-desc">Average absolute deviation</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-title">R² Score (R-squared)</div>
                            <div class="kpi-value">{m['R2']:.4f}</div>
                            <div class="kpi-desc">Variance explained (Max 1.0)</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-title">Mean Squared Error (MSE)</div>
                            <div class="kpi-value">{m['MSE']:.2f}</div>
                            <div class="kpi-desc">Average squared deviation</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns(2)
            for idx, model_key in enumerate(metrics.keys()):
                m = metrics[model_key]
                with (col1 if idx == 0 else col2):
                    st.markdown(f"<h4 style='text-align: center; color: #ffffff;'>{model_key} Performance Metrics</h4>", unsafe_allow_html=True)
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.markdown(f"""
                            <div class="kpi-card" style="margin-bottom: 10px;">
                                <div class="kpi-title">RMSE</div>
                                <div class="kpi-value">${m['RMSE']:.2f}</div>
                            </div>
                            <div class="kpi-card">
                                <div class="kpi-title">R² Score</div>
                                <div class="kpi-value">{m['R2']:.4f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with mcol2:
                        st.markdown(f"""
                            <div class="kpi-card" style="margin-bottom: 10px;">
                                <div class="kpi-title">MAE</div>
                                <div class="kpi-value">${m['MAE']:.2f}</div>
                            </div>
                            <div class="kpi-card">
                                <div class="kpi-title">MSE</div>
                                <div class="kpi-value">{m['MSE']:.2f}</div>
                            </div>
                        """, unsafe_allow_html=True)

        st.write("")
        st.write("")

        st.subheader("Test Data Predictions vs Actual Prices")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=test_dates, 
            y=y_test_inv.flatten(),
            mode='lines',
            name='Actual TSLA Close Price',
            line=dict(color='#8b949e', width=2),
            hoverlabel=dict(namelength=-1)
        ))

        colors = {"Simple RNN": "#fbbf24", "LSTM": "#3b82f6"}
        for model_key, pred in predictions.items():
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=pred.flatten(),
                mode='lines',
                name=f'{model_key} Prediction',
                line=dict(color=colors.get(model_key, '#3b82f6'), width=2, dash='dash'),
                hoverlabel=dict(namelength=-1)
            ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                title=dict(text="Date", font=dict(color="#8b949e")),
                tickfont=dict(color="#8b949e")
            ),
            yaxis=dict(
                showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                title=dict(text="Stock Price (USD)", font=dict(color="#8b949e")),
                tickfont=dict(color="#8b949e")
            ),
            legend=dict(
                font=dict(color="#8b949e"),
                bgcolor='rgba(0,0,0,0)',
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=40, r=40, t=50, b=40),
            height=550
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Autoregressive Future Price Forecasting")
        st.markdown(f"""
            This section forecasts the stock price of Tesla for the next **{forecast_days} trading days** by feeding the model's predictions back into itself iteratively.
        """)

        forecast_model_choice = st.selectbox(
            "Select Forecasting Model",
            options=["LSTM", "Simple RNN"] if (lstm_model is not None and rnn_model is not None) else (["LSTM"] if lstm_model is not None else ["Simple RNN"])
        )

        active_model = lstm_model if forecast_model_choice == "LSTM" else rnn_model

        if active_model is not None:
            last_sequence = scaled_data[-time_step:]
            current_sequence = last_sequence.reshape(1, time_step, 1)
            
            future_predictions_scaled = []
            
            for _ in range(forecast_days):
                next_pred = active_model.predict(current_sequence, verbose=0)
                future_predictions_scaled.append(next_pred[0, 0])
                
                new_step = next_pred.reshape(1, 1, 1)
                current_sequence = np.append(current_sequence[:, 1:, :], new_step, axis=1)

            future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')

            hist_days_to_show = 120
            hist_subset = df.iloc[-hist_days_to_show:]
            
            fig_fc = go.Figure()

            fig_fc.add_trace(go.Scatter(
                x=hist_subset.index, 
                y=hist_subset["Close"],
                mode='lines',
                name='Historical Close Price (Last 120 Days)',
                line=dict(color='#58a6ff', width=2),
                hoverlabel=dict(namelength=-1)
            ))

            fc_dates_plot = [hist_subset.index[-1]] + list(future_dates)
            fc_values_plot = [hist_subset["Close"].iloc[-1]] + list(future_predictions.flatten())

            fig_fc.add_trace(go.Scatter(
                x=fc_dates_plot, 
                y=fc_values_plot,
                mode='lines+markers',
                name=f'{forecast_model_choice} Future Forecast ({forecast_days} Days)',
                line=dict(color='#ff7b72', width=2, dash='dot'),
                marker=dict(size=4),
                hoverlabel=dict(namelength=-1)
            ))

            fig_fc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                    title=dict(text="Date", font=dict(color="#8b949e")),
                    tickfont=dict(color="#8b949e")
                ),
                yaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                    title=dict(text="Stock Price (USD)", font=dict(color="#8b949e")),
                    tickfont=dict(color="#8b949e")
                ),
                legend=dict(
                    font=dict(color="#8b949e"),
                    bgcolor='rgba(0,0,0,0)',
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=40, r=40, t=50, b=40),
                height=550
            )

            st.plotly_chart(fig_fc, use_container_width=True)

            st.subheader("Forecasted Price List")
            fc_df = pd.DataFrame(
                data=future_predictions,
                index=future_dates,
                columns=["Forecasted Close Price (USD)"]
            )
            fc_df.index.name = "Date"
            
            st.dataframe(
                fc_df.style.format("${:.2f}").background_gradient(cmap="Blues"),
                use_container_width=True
            )
        else:
            st.warning("Model could not be loaded. Please run 'main.py' to train models first.")

    with tab3:
        st.subheader("Tesla Dataset Overview")
        
        latest_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        change = latest_row["Close"] - prev_row["Close"]
        pct_change = (change / prev_row["Close"]) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Latest Close Price", 
                value=f"${latest_row['Close']:.2f}",
                delta=f"${change:.2f} ({pct_change:.2f}%)"
            )
        with col2:
            st.metric(
                label="Highest Price (Dataset)", 
                value=f"${df['High'].max():.2f}"
            )
        with col3:
            st.metric(
                label="Lowest Price (Dataset)", 
                value=f"${df['Low'].min():.2f}"
            )
        with col4:
            st.metric(
                label="Total Trading Days", 
                value=f"{len(df):,}"
            )

        st.write("")
        st.subheader("Data Head & Tail View")
        
        sub_tab_1, sub_tab_2 = st.tabs(["📋 Data Preview", "📊 Statistical Summary"])
        
        with sub_tab_1:
            st.dataframe(df, use_container_width=True)
            
        with sub_tab_2:
            st.dataframe(df.describe(), use_container_width=True)

        st.write("")
        st.subheader("Trading Volume Analysis")
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker=dict(color="#3b82f6", opacity=0.6)
        ))
        fig_vol.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color="#8b949e")),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Volume", tickfont=dict(color="#8b949e")),
            margin=dict(l=40, r=40, t=10, b=40),
            height=300
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab4:
        st.subheader("Understanding Deep Learning Models for Time-Series")
        
        col_rnn, col_lstm = st.columns(2)
        
        with col_rnn:
            st.markdown("""
                <div style="background: rgba(251, 191, 36, 0.05); border: 1px solid rgba(251, 191, 36, 0.2); border-radius: 12px; padding: 20px; height: 100%;">
                    <h3 style="color: #fbbf24; margin-top:0;">Simple Recurrent Neural Network (RNN)</h3>
                    <p style="font-size: 14px; color: #e2e8f0;">
                        Simple RNNs process sequences by passing their previous hidden state to the next step along with the current input.
                    </p>
                    <h5 style="color: #fbbf24;">Characteristics:</h5>
                    <ul style="font-size: 13px; color: #94a3b8;">
                        <li>Fast to train and execute</li>
                        <li>Effective for shorter sequences and simple patterns</li>
                        <li>Prone to <strong>Vanishing Gradients</strong> (struggles with long-term memory)</li>
                    </ul>
                    <h5 style="color: #fbbf24;">Architecture in use:</h5>
                    <code style="display: block; background: #0f172a; padding: 10px; border-radius: 6px; color: #fbbf24; font-size:12px;">
                        Sequential(<br>
                        &nbsp;&nbsp;SimpleRNN(units=50, input_shape=(60, 1)),<br>
                        &nbsp;&nbsp;Dropout(rate=0.2),<br>
                        &nbsp;&nbsp;Dense(units=1)<br>
                        )
                    </code>
                </div>
            """, unsafe_allow_html=True)
            
        with col_lstm:
            st.markdown("""
                <div style="background: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 20px; height: 100%;">
                    <h3 style="color: #3b82f6; margin-top:0;">Long Short-Term Memory (LSTM)</h3>
                    <p style="font-size: 14px; color: #e2e8f0;">
                        LSTMs address the vanishing gradient problem by introducing a <strong>cell state</strong> and gate mechanisms (input, forget, output gates).
                    </p>
                    <h5 style="color: #3b82f6;">Characteristics:</h5>
                    <ul style="font-size: 13px; color: #94a3b8;">
                        <li>Excellent at capturing long-term dependencies and trends</li>
                        <li>Robust against vanishing gradient issues</li>
                        <li>Slightly slower training times due to complex architecture</li>
                    </ul>
                    <h5 style="color: #3b82f6;">Architecture in use:</h5>
                    <code style="display: block; background: #0f172a; padding: 10px; border-radius: 6px; color: #3b82f6; font-size:12px;">
                        Sequential(<br>
                        &nbsp;&nbsp;LSTM(units=64, return_sequences=True, input_shape=(60, 1)),<br>
                        &nbsp;&nbsp;Dropout(rate=0.2),<br>
                        &nbsp;&nbsp;LSTM(units=64),<br>
                        &nbsp;&nbsp;Dropout(rate=0.2),<br>
                        &nbsp;&nbsp;Dense(units=1)<br>
                        )
                    </code>
                </div>
            """, unsafe_allow_html=True)
else:
    st.error("No dataset available to perform prediction analytics.")
