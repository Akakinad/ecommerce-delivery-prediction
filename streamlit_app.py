# streamlit_app.py
import streamlit as st
st.cache_data.clear()  # Clear all caches on startup
import pandas as pd
import joblib
import os

# --- Page setup ---
st.set_page_config(page_title="📦 E-Commerce Delivery Predictor", layout="wide")
st.title("📦 Predict E-Commerce Delivery Delays")
st.write("""
Use this interactive app to predict whether an electronics order will arrive **on time** or **late**.
Adjust the order details on the sidebar and see the prediction instantly.
""")

# --- Load models and encoders ---
try:
    models_folder = "models"
    log_reg = joblib.load(os.path.join(models_folder, "log_reg.pkl"))
    best_rf = joblib.load(os.path.join(models_folder, "best_rf.pkl"))
    ordinal_encoder = joblib.load(os.path.join(models_folder, "ordinal_encoder.pkl"))
    scaler = joblib.load(os.path.join(models_folder, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(models_folder, "feature_names.pkl"))
    
    st.sidebar.success(f"✅ Models loaded ({len(feature_names)} features)")
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Order Features")

weight_in_gms = st.sidebar.number_input("Product Weight (grams)", min_value=0, value=5000, step=100)
cost_of_the_product = st.sidebar.number_input("Product Cost ($)", min_value=0, value=200, step=1)
discount = st.sidebar.number_input("Discount ($)", min_value=0, value=10, step=1)
customer_care_calls = st.sidebar.number_input("Customer Support Calls", min_value=0, value=2, step=1)
customer_rating = st.sidebar.slider("Customer Rating (1-5)", min_value=1, max_value=5, value=3)
prior_purchases = st.sidebar.number_input("Prior Purchases", min_value=0, value=3, step=1)
warehouse_block = st.sidebar.selectbox("Warehouse Location", ["A", "B", "C", "D", "F"])
mode_of_shipment = st.sidebar.selectbox("Mode of Shipment", ["Flight", "Road", "Ship"])
gender = st.sidebar.selectbox("Customer Gender", ["F", "M"])
product_importance = st.sidebar.selectbox("Product Importance", ["low", "medium", "high"])
model_choice = st.sidebar.radio("Choose Model", ("Logistic Regression", "Random Forest"))

# --- Feature Engineering ---
# Discount bin
if discount < 20:
    discount_bin = "Low"
elif discount < 50:
    discount_bin = "Medium"
else:
    discount_bin = "High"

# Engineered features
cost_weight_interaction = cost_of_the_product * weight_in_gms
calls_per_prior_purchase = customer_care_calls / (prior_purchases + 1)

# --- Create input dataframe ---
input_data = {
    "customer_care_calls": customer_care_calls,
    "customer_rating": customer_rating,
    "cost_of_the_product": cost_of_the_product,
    "prior_purchases": prior_purchases,
    "product_importance": product_importance,
    "weight_in_gms": weight_in_gms,
    "discount_bin": discount_bin,
    "warehouse_block": warehouse_block,
    "mode_of_shipment": mode_of_shipment,
    "gender": gender,
    "cost_weight_interaction": cost_weight_interaction,
    "calls_per_prior_purchase": calls_per_prior_purchase
}

input_df = pd.DataFrame([input_data])

# --- Encode ordinal features ---
ordinal_cols = ['product_importance', 'discount_bin']
input_df[ordinal_cols] = ordinal_encoder.transform(input_df[ordinal_cols])

# --- One-hot encode categorical features ---
cat_cols = ["warehouse_block", "mode_of_shipment", "gender"]
input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=False, dtype=int)

# --- Reindex to match training columns exactly ---
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# --- Predict ---
try:
    if model_choice == "Logistic Regression":
        # Scale the input for Logistic Regression
        input_scaled = scaler.transform(input_df)
        prediction = log_reg.predict(input_scaled)[0]
        prediction_proba = log_reg.predict_proba(input_scaled)[0][1]
    else:
        # Random Forest doesn't need scaling
        prediction = best_rf.predict(input_df)[0]
        prediction_proba = best_rf.predict_proba(input_df)[0][1]
    
    # --- Display results ---
    st.subheader(f"Prediction Result ({model_choice})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error(f"⚠️ **LATE DELIVERY PREDICTED**")
            st.metric("Probability of Delay", f"{prediction_proba*100:.1f}%")
        else:
            st.success(f"✅ **ON-TIME DELIVERY PREDICTED**")
            st.metric("Probability of On-Time", f"{(1-prediction_proba)*100:.1f}%")
    
    with col2:
        st.info("**Risk Factors:**")
        risk_factors = []
        if weight_in_gms > 4000:
            risk_factors.append("⚖️ Heavy product")
        if discount > 40:
            risk_factors.append("🏷️ High discount")
        if customer_care_calls > 3:
            risk_factors.append("📞 Multiple support calls")
        if cost_of_the_product > 250:
            risk_factors.append("💰 High-value item")
        if calls_per_prior_purchase > 1.5:
            risk_factors.append("🚨 High call-to-purchase ratio")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.write("✅ No major risk factors detected")
        
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.write("**Debug Info:**")
    st.write(f"Input shape: {input_df.shape}")
    st.write(f"Expected features: {len(feature_names)}")
    with st.expander("See column details"):
        st.write("Input columns:", input_df.columns.tolist())
        st.write("\nExpected columns:", feature_names)

# --- Model Performance Metrics ---
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.metric("Recall (Late Deliveries)", "87%")
    st.metric("Accuracy", "82%")

with col2:
    st.metric("F1-Score", "0.84")
    st.metric("Total Training Samples", "10,999")

# --- Model Insights ---
st.subheader("🔍 Model Insights")
col1, col2 = st.columns(2)

with col1:
    if os.path.exists("images/confusion_matrix.png"):
        st.image("images/confusion_matrix.png", caption="Confusion Matrix")
    else:
        st.info("💡 Add confusion_matrix.png to /images folder to display")

with col2:
    if os.path.exists("images/feature_importance.png"):
        st.image("images/feature_importance.png", caption="Feature Importance")
    else:
        st.info("💡 Add feature_importance.png to /images folder to display")

# --- Show raw input for transparency ---
with st.expander("🔧 See processed input features"):
    #st.dataframe(input_df, use_container_width=True)
    st.dataframe(input_df, width='stretch')


# --- Footer ---
st.markdown("---")
st.markdown("**Built with:** Python • Scikit-learn • Streamlit")

# ============================================================================
# FOOTER WITH VISITOR COUNTER
# ============================================================================

st.markdown("---")

# Visitor counter using hits.sh
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style='text-align: center;'>
        <img src='https://hits.sh/github.com/Akakinad/ecommerce-delivery-prediction.svg?label=Visitors&color=4c1&labelColor=2c3e50' alt='Visitors'/>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>Built with Streamlit 🎈 | Powered by Machine Learning 🤖</p>
    <p>This is a portfolio project demonstrating end-to-end ML workflow</p>
</div>
""", unsafe_allow_html=True)
