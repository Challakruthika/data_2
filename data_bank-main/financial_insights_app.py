import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from prophet import Prophet
from sklearn.ensemble import IsolationForest

# --- Categorization ---
# Load the trained ML model
model = joblib.load("category_model.pkl")

def categorize(desc, amt):
    try:
        return model.predict([str(desc)])[0]
    except Exception:
        return 'Salary/Income' if amt > 0 else 'Others'

# --- Streamlit Setup ---
st.set_page_config(page_title="💰 Financial Insights Dashboard", layout="wide")
st.title("💸 Financial Insights Dashboard")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your bank statement (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Detect date, amount, and description columns
    possible_date_cols = [col for col in data.columns if 'date' in col.lower()]
    possible_amount_cols = [col for col in data.columns if 'amount' in col.lower()]
    possible_desc_cols = [col for col in data.columns if 'desc' in col.lower() or 'narration' in col.lower() or 'particular' in col.lower()]

    date_col = possible_date_cols[0] if possible_date_cols else "Date"
    amount_col = possible_amount_cols[0] if possible_amount_cols else "Amount"
    desc_col = possible_desc_cols[0] if possible_desc_cols else "None"

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col])
    data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce')
    data = data.dropna(subset=[amount_col])

    # --- Categorization ---
    if desc_col != "None":
        data['category'] = data.apply(lambda row: categorize(row[desc_col], row[amount_col]), axis=1)
    else:
        data['category'] = data[amount_col].apply(lambda amt: 'Salary/Income' if amt > 0 else 'Others')

    # Infer flow direction (income vs expense)
    data["Flow"] = data.apply(lambda row: row[amount_col] if row["category"] in ["Income", "Salary", "Salary/Income"] else -row[amount_col], axis=1)
    data["Month"] = data[date_col].dt.to_period("M").astype(str)

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📁 Categories", "🔮 Forecast", "🚨 Anomalies", "📥 Download"])

    with tab1:
        st.subheader("📈 Monthly Net Flow")
        monthly_net = data.groupby("Month")["Flow"].sum()
        st.bar_chart(monthly_net)

    with tab2:
        st.subheader("📂 Category-wise Spend/Income")
        category_summary = data.groupby("category")["Flow"].sum().sort_values()
        fig, ax = plt.subplots()
        sns.barplot(x=category_summary.values, y=category_summary.index, palette="magma", ax=ax)
        ax.set_title("Total by Category")
        st.pyplot(fig)

    with tab3:
        st.subheader("🔮 Net Flow Forecast (Next 6 Months)")
        try:
            df_prophet = data.groupby("Month")["Flow"].sum().reset_index()
            df_prophet.columns = ["ds", "y"]
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
            model_prophet = Prophet()
            model_prophet.fit(df_prophet)
            future = model_prophet.make_future_dataframe(periods=6, freq="M")
            forecast = model_prophet.predict(future)
            fig1 = model_prophet.plot(forecast)
            st.pyplot(fig1)
        except Exception as e:
            st.warning("⚠️ Forecasting failed. Check for valid data.")
            st.text(str(e))

    with tab4:
        st.subheader("🚨 Anomalous Transactions")
        iso = IsolationForest(contamination=0.01, random_state=42)
        data["anomaly"] = iso.fit_predict(data[[amount_col]])
        anomalies = data[data["anomaly"] == -1]
        st.write(f"Found {len(anomalies)} anomalous transactions.")
        st.dataframe(anomalies[[date_col, amount_col, "category"] + ([desc_col] if desc_col != "None" else [])])

    with tab5:
        st.subheader("📥 Download Categorized Data")
        st.dataframe(data.head(10))
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "categorized_data.csv", "text/csv")

