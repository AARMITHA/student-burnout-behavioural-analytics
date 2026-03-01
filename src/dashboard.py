import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Student Burnout Risk System",
    page_icon="🎓",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }
    .metric-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
    }
    .title-text {
        font-size: 36px;
        font-weight: 700;
        color: #2c3e50;
    }
    .subtitle-text {
        font-size: 18px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

df = pd.read_csv("../data/student_risk_scores.csv")

st.markdown('<div class="title-text">Student Burnout Risk Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Early identification of academic burnout using machine learning</div>', unsafe_allow_html=True)
st.write("")

low_count = (df["risk_category"] == "Low").sum()
medium_count = (df["risk_category"] == "Medium").sum()
high_count = (df["risk_category"] == "High").sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'<div class="metric-box"><h2>{low_count}</h2><p>Low Risk Students</p></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="metric-box"><h2>{medium_count}</h2><p>Medium Risk Students</p></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="metric-box"><h2>{high_count}</h2><p>High Risk Students</p></div>', unsafe_allow_html=True)

st.write("")
st.write("")

st.subheader("Burnout Risk Distribution")
st.bar_chart(df["risk_category"].value_counts())

st.write("")
st.subheader("Filter Students")

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    risk_filter = st.selectbox("Select Risk Category", ["All", "Low", "Medium", "High"])

with filter_col2:
    max_prob = st.slider("Maximum Burnout Probability", 0.0, 1.0, 1.0)

filtered_df = df.copy()

if risk_filter != "All":
    filtered_df = filtered_df[filtered_df["risk_category"] == risk_filter]

filtered_df = filtered_df[filtered_df["burnout_probability"] <= max_prob]

st.write("")
st.dataframe(filtered_df, use_container_width=True, height=350)

st.write("")
st.subheader("Individual Student Risk Analysis")

student_id = st.number_input(
    "Select Student Index",
    min_value=0,
    max_value=len(df) - 1,
    step=1
)

student = df.iloc[int(student_id)]

st.markdown(
    f"""
    <div class="metric-box">
        <h3>Burnout Probability</h3>
        <h1>{student['burnout_probability']:.3f}</h1>
        <p><b>Risk Level:</b> {student['risk_category']}</p>
        <p><b>Explanation:</b> {student['risk_explanation']}</p>
    </div>
    """,
    unsafe_allow_html=True
)