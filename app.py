import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon=" ",
    layout="wide"
)

st.title(" Customer Churn Prediction")
st.write("Masukkan data pelanggan untuk memprediksi apakah pelanggan akan churn.")

model = joblib.load("model_churn_diva.pkl")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox("Gender", ["Male","Female"])

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    country = st.text_input("Country")

    city = st.text_input("City")

    acquisition_channel = st.text_input("Acquisition Channel")

    device_type = st.text_input("Device Type")

    subscription_type = st.text_input("Subscription Type")

    is_premium_user = st.selectbox(
        "Premium User",
        [0,1]
    )

    total_visits = st.number_input(
        "Total Visits",
        value=0
    )

    avg_session_time = st.number_input(
        "Average Session Time",
        value=0.0
    )

    pages_per_session = st.number_input(
        "Pages per Session",
        value=0.0
    )

with col2:

    email_open_rate = st.number_input(
        "Email Open Rate",
        value=0.0
    )

    email_click_rate = st.number_input(
        "Email Click Rate",
        value=0.0
    )

    total_spent = st.number_input(
        "Total Spent",
        value=0.0
    )

    avg_order_value = st.number_input(
        "Average Order Value",
        value=0.0
    )

    discount_used = st.number_input(
        "Discount Used",
        value=0
    )

    support_tickets = st.number_input(
        "Support Tickets",
        value=0
    )

    refund_requested = st.selectbox(
        "Refund Requested",
        [0,1]
    )

    delivery_delay_days = st.number_input(
        "Delivery Delay Days",
        value=0
    )

    payment_method = st.text_input("Payment Method")

    satisfaction_score = st.number_input(
        "Satisfaction Score",
        value=5
    )

    nps_score = st.number_input(
        "NPS Score",
        value=0
    )

    marketing_spend_per_user = st.number_input(
        "Marketing Spend Per User",
        value=0.0
    )

    lifetime_value = st.number_input(
        "Lifetime Value",
        value=0.0
    )

    last_3_month_purchase_freq = st.number_input(
        "Last 3 Month Purchase Frequency",
        value=0
    )

if st.button(" Predict"):

    data = pd.DataFrame({

        "gender":[gender],
        "age":[age],
        "country":[country],
        "city":[city],
        "acquisition_channel":[acquisition_channel],
        "device_type":[device_type],
        "subscription_type":[subscription_type],
        "is_premium_user":[is_premium_user],
        "total_visits":[total_visits],
        "avg_session_time":[avg_session_time],
        "pages_per_session":[pages_per_session],
        "email_open_rate":[email_open_rate],
        "email_click_rate":[email_click_rate],
        "total_spent":[total_spent],
        "avg_order_value":[avg_order_value],
        "discount_used":[discount_used],
        "support_tickets":[support_tickets],
        "refund_requested":[refund_requested],
        "delivery_delay_days":[delivery_delay_days],
        "payment_method":[payment_method],
        "satisfaction_score":[satisfaction_score],
        "nps_score":[nps_score],
        "marketing_spend_per_user":[marketing_spend_per_user],
        "lifetime_value":[lifetime_value],
        "last_3_month_purchase_freq":[last_3_month_purchase_freq]

    })

    hasil = model.predict(data)

    if hasil[0] == 1:
        st.error("⚠ Customer diprediksi akan CHURN")
    else:
        st.success(" Customer diprediksi TIDAK CHURN")