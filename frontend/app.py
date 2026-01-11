import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #f9fafb;
}
h1, h2, h3 {
    color: #1f2937;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
}
.stProgress > div > div > div {
    background-color: #22c55e;
}
.metric-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    text-align: center;
    color: #000000;   
}
.metric-card h2, 
.metric-card h3 {
    color: #000000;   
}

</style>
""", unsafe_allow_html=True)

# =========================
# AUTHENTICATION
# =========================
def login():
    st.title("🔐 Shopkeeper Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# =========================
# MAIN APP
# =========================
def main_app():

    # Logout
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("""
    <h1>🛒 Customer Purchase Prediction System</h1>
    <p style='font-size:18px;color:#374151'>
    AI-powered decision support tool for shopkeepers
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # =========================
    # LOAD MODEL
    # =========================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = pickle.load(open(os.path.join(BASE_DIR, "models", "best_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb"))

    # =========================
    # SIDEBAR INPUTS
    # =========================
    st.sidebar.header("🧾 Customer Details")

    age = st.sidebar.number_input("Customer Age", 18, 100, 30)
    income = st.sidebar.number_input("Annual Income", 10000, 200000, 50000)
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    children = st.sidebar.number_input("Number of Children", 0, 5, 0)
    shopping_score = st.sidebar.slider("Shopping History Score", 1, 100, 50)
    product = st.sidebar.selectbox("Product Category Interest",
                                   ["Electronics", "Fashion", "Home Decor", "Sports"])
    discount = st.sidebar.slider("Discount Offered (%)", 0, 50, 10)

    # =========================
    # ENCODING
    # =========================
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
    education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    product_map = {"Electronics": 0, "Fashion": 1, "Home Decor": 2, "Sports": 3}

    input_data = np.array([[
        age,
        income,
        marital_map[marital_status],
        education_map[education],
        children,
        shopping_score,
        product_map[product],
        discount
    ]])

    # Initialize history
    if "history" not in st.session_state:
        st.session_state.history = []

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📜 History", "ℹ About"])

    # ---------- PREDICT TAB ----------
    with tab1:
        if st.button("🔍 Predict Purchase"):
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                    <h3>Purchase Probability</h3>
                    <h2>{probability*100:.2f}%</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                    <h3>Prediction</h3>
                    <h2>{"YES" if prediction==1 else "NO"}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.progress(int(probability * 100))

            if prediction == 1:
                st.success("✅ High chance of purchase — offer personalized deals!")
            else:
                st.warning("⚠ Low chance — consider discount or engagement offer.")

            st.session_state.history.append({
                "Age": age,
                "Income": income,
                "Product": product,
                "Discount": discount,
                "Prediction": "Yes" if prediction == 1 else "No",
                "Probability (%)": round(probability * 100, 2)
            })

    # ---------- HISTORY TAB ----------
    with tab2:
        st.subheader("🧾 Prediction History")
        if st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True)
            st.download_button("⬇ Download History CSV",
                               df_history.to_csv(index=False),
                               "prediction_history.csv",
                               "text/csv")
        else:
            st.info("No predictions made yet.")

    # ---------- ABOUT TAB ----------
    with tab3:
        st.subheader("ℹ About This System")
        st.markdown("""
        **Customer Purchase Prediction System** helps shopkeepers:
        - Predict purchase likelihood  
        - Improve sales strategy  
        - Offer personalized discounts  

        **Tech Stack**
        - Python, Scikit-learn
        - Streamlit
        - Pandas, NumPy
        """)

    st.markdown("<hr><center>🚀 Built with Streamlit | ML Project</center>",
                unsafe_allow_html=True)

# =========================
# APP FLOW
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login()
