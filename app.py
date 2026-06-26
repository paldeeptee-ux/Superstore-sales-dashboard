import streamlit as st
import pandas as pd
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Python Dashboard", layout="wide")

st.title("📊 Data Analysis Dashboard")

# Load Dataset
df = pd.read_csv("train(1).csv")

# Show Dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset Information
st.subheader("Dataset Information")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

# Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Numeric Columns
numeric_cols = df.select_dtypes(include=['int64','float64']).columns

if len(numeric_cols) > 0:

    # Select Column
    column = st.selectbox("Select Numeric Column", numeric_cols)

    # Histogram
    fig = px.histogram(df, x=column, title=f"Distribution of {column}")
    st.plotly_chart(fig, use_container_width=True)

    # Box Plot
    fig2 = px.box(df, y=column, title=f"Box Plot of {column}")
    st.plotly_chart(fig2, use_container_width=True)

# Categorical Columns
cat_cols = df.select_dtypes(include=['object']).columns

if len(cat_cols) > 0:

    cat = st.selectbox("Select Category Column", cat_cols)

    chart = df[cat].value_counts().reset_index()
    chart.columns = [cat, "Count"]

    fig3 = px.bar(chart,
                  x=cat,
                  y="Count",
                  title=f"{cat} Count")
    st.plotly_chart(fig3, use_container_width=True)

# Correlation Heatmap
if len(numeric_cols) > 1:

    st.subheader("Correlation Matrix")

    corr = df[numeric_cols].corr()

    fig4 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig4, use_container_width=True)

# Summary Statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Download Clean Data
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "clean_data.csv",
    "text/csv"
)