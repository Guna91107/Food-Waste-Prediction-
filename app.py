import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Food Waste Prediction System",
    page_icon="🍽️",
    layout="wide"
)

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #F6FAF9 0%, #EEF5F2 100%);
}

h1 {
    font-weight: 800;
    letter-spacing: 0.5px;
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #1F2937, #111827);
    color: white;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

div[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
}

div[data-testid="metric-container"] > label {
    font-size: 14px;
    color: #6B7280 !important;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

with open("model/rf_model.pkl", "rb") as file:
    model = pickle.load(file)

st.sidebar.markdown("## 🍽️ Food Planning Inputs")
st.sidebar.markdown("Enter expected planning details")

food_prepared = st.sidebar.number_input(
    "Food Prepared (Units / Portions)",
    min_value=50.0,
    step=10.0
)

people_served = st.sidebar.number_input(
    "Expected Number of People",
    min_value=50,
    step=10
)

day_type = st.sidebar.selectbox(
    "Day Type",
    ["Weekday", "Weekend"]
)

event_day = st.sidebar.selectbox(
    "Event Day",
    ["No", "Yes"]
)

predict_btn = st.sidebar.button("🔍 Predict Food Waste")

day_type_val = 0 if day_type == "Weekday" else 1
event_day_val = 0 if event_day == "No" else 1

st.markdown("---")

if predict_btn:

    if food_prepared < people_served:
        st.error(
            "❌ Food shortage detected.\n\n"
            "Prepared food is less than expected demand.\n\n"
            "Prediction not applicable."
        )

    elif food_prepared < people_served * 0.3:
        st.warning(
            "⚠️ Food prepared is unrealistically low compared to people served."
        )

    else:
        input_data = np.array([[food_prepared, people_served, day_type_val, event_day_val]])
        prediction = model.predict(input_data)[0]

        demand_ratio = people_served / food_prepared

        if demand_ratio < 1:
            prediction = prediction * (1 + (1 - demand_ratio))
        else:
            prediction = prediction * 0.9

        if event_day == "Yes":
            prediction = prediction * 1.08  

        if day_type == "Weekend":
            prediction = prediction * 1.05 

        prediction = min(prediction, food_prepared)
        prediction = max(0, prediction)

        col1, col2, col3 = st.columns(3)

        col1.metric("🍽️ Predicted Food Waste", f"{prediction:.2f} units")

        waste_percentage = (prediction / food_prepared) * 100
        col2.metric("📉 Waste Percentage", f"{waste_percentage:.2f}%")

        cost_loss = prediction * 50
        col3.metric("💰 Estimated Cost Loss", f"₹ {cost_loss:,.0f}")

        st.markdown("---")

        st.subheader("📊 Food Planning Comparison")

        df = pd.DataFrame({
            "Category": ["Food Prepared", "People Served", "Food Wasted"],
            "Value": [food_prepared, people_served, prediction]
        })

        st.bar_chart(df.set_index("Category"))

        st.subheader("📊 Food Utilization Distribution")

        food_used = food_prepared - prediction
        pie_data = [food_used, prediction]
        pie_labels = ["Food Consumed", "Food Wasted"]

        fig, ax = plt.subplots()
        ax.pie(
            pie_data,
            labels=pie_labels,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white'}
        )
        ax.axis('equal')

        st.pyplot(fig)

        st.markdown("---")

        st.subheader("🔍 Reason Analysis")

        reasons = []
        main_reason = ""

        if food_prepared > people_served * 1.25:
            reasons.append("Over-preparation compared to demand")
            main_reason = "Over-preparation"

        if people_served < food_prepared * 0.75:
            reasons.append("Lower-than-expected demand")
            if not main_reason:
                main_reason = "Low demand"

        if event_day == "Yes":
            reasons.append("Event day caused demand uncertainty")
            if not main_reason:
                main_reason = "Event uncertainty"

        if prediction < 20:
            reasons.append("Balanced food planning")
            main_reason = "Good planning"

        for r in reasons:
            st.write("•", r)

        st.markdown(f"**Main Reason:** `{main_reason}`")

st.markdown("---")