import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Customer Churn Prediction")
st.write("Masukkan data pelanggan di bawah ini untuk memprediksi probabilitas atau status *churn*.")
st.markdown("---")

# Load Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load("model_churn_diva.pkl")

model = load_model()

# Membuat Form Input dengan 2 Kolom Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Profil & Demografi Pelanggan")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    country = st.text_input("Country", value="Indonesia")
    city = st.text_input("City", value="Jakarta")
    acquisition_channel = st.text_input("Acquisition Channel", value="Organic")
    device_type = st.text_input("Device Type", value="Mobile")
    subscription_type = st.text_input("Subscription Type", value="Monthly")
    is_premium_user = st.selectbox("Premium User", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    payment_method = st.text_input("Payment Method", value="Credit Card")

with col2:
    st.subheader("📊 Aktivitas & Metrik Penggunaan")
    total_visits = st.number_input("Total Visits", min_value=0, value=10)
    avg_session_time = st.number_input("Average Session Time (minutes)", min_value=0.0, value=15.5)
    pages_per_session = st.number_input("Pages Per Session", min_value=0.0, value=4.0)
    email_open_rate = st.number_input("Email Open Rate (%)", min_value=0.0, max_value=100.0, value=25.0)
    email_click_rate = st.number_input("Email Click Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
    total_spent = st.number_input("Total Spent", min_value=0.0, value=500000.0)
    avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=50000.0)
    discount_used = st.selectbox("Discount Used", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    support_tickets = st.number_input("Support Tickets", min_value=0, value=0)
    refund_requested = st.selectbox("Refund Requested", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    delivery_delay_days = st.number_input("Delivery Delay Days", min_value=0, value=0)
    satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=4)
    nps_score = st.slider("NPS Score", min_value=1, max_value=10, value=8)
    marketing_spend_per_user = st.number_input("Marketing Spend Per User", min_value=0.0, value=15000.0)
    last_3_month_purchase_freq = st.number_input("Last 3 Month Purchase Frequency", min_value=0, value=2)

st.markdown("---")

# Tombol Prediksi
if st.button("🚀 Predict Churn Status", use_container_width=True):
    
    # 1. Bungkus input ke dalam DataFrame dengan key sesuai fitur asli model
    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "country": [country],
        "city": [city],
        "acquisition_channel": [acquisition_channel],
        "device_type": [device_type],
        "subscription_type": [subscription_type],
        "is_premium_user": [is_premium_user],
        "total_visits": [total_visits],
        "avg_session_time": [avg_session_time],
        "pages_per_session": [pages_per_session],
        "email_open_rate": [email_open_rate],
        "email_click_rate": [email_click_rate],
        "total_spent": [total_spent],
        "avg_order_value": [avg_order_value],
        "discount_used": [discount_used],
        "support_tickets": [support_tickets],
        "refund_requested": [refund_requested],
        "delivery_delay_days": [delivery_delay_days],
        "payment_method": [payment_method],
        "satisfaction_score": [satisfaction_score],
        "nps_score": [nps_score],
        "marketing_spend_per_user": [marketing_spend_per_user],
        "last_3_month_purchase_freq": [last_3_month_purchase_freq]
    })

    try:
        # 2. Lakukan Prediksi
        prediction = model.predict(input_data)
        
        # Mendapatkan probabilitas jika model mendukung predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]
        else:
            proba = None

        # 3. Tampilkan Hasil Ke Layar
        st.subheader("🔮 Hasil Analisis Prediksi:")
        
        if prediction[0] == 1:
            st.error("🚨 **Pelanggan diprediksi akan CHURN!**")
            if proba is not None:
                st.write(f"Probabilitas Churn: **{proba * 100:.2f}%**")
            st.info("💡 **Rekomendasi:** Berikan penawaran khusus atau diskon retensi untuk mempertahankan pelanggan ini.")
        else:
            st.success("✅ **Pelanggan diprediksi TETAP SETIA (Not Churn)!**")
            if proba is not None:
                st.write(f"Probabilitas Tetap Setia: **{(1 - proba) * 100:.2f}%**")
                
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        st.warning("Pastikan urutan atau tipe data kolom input sesuai dengan format training pada model pickle Anda.")
