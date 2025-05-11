import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Music & Mental Health", layout="wide")
st.title("🎧 Does Music Help Mental Health?")

# Load your dataset
df = pd.read_csv("mxmh_survey_results.csv")

# Clean columns
df.columns = [col.strip() for col in df.columns]

# Convert numeric columns
num_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Hours per day', 'BPM']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing mental health or listening hours
df = df.dropna(subset=['Hours per day', 'Anxiety', 'Depression', 'Insomnia', 'OCD'])

# Sidebar filters
st.sidebar.header("Filter Data")
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (18, 65))
genres = st.sidebar.multiselect("Favorite Genres", df["Fav genre"].dropna().unique(), default=df["Fav genre"].dropna().unique())

filtered_df = df[(df["Age"] >= age_range[0]) &
                 (df["Age"] <= age_range[1]) &
                 (df["Fav genre"].isin(genres))]

import plotly.express as px

# SECTION 1: Scatter plot - Hours per day vs. Mental Health metric by Genre
st.subheader("Does Listening Time Affect Mental Health? (By Genre)")

metric = st.selectbox("Choose a mental health metric", ['Anxiety', 'Depression', 'Insomnia', 'OCD'])

# Filter data for selected metric and drop rows with missing values
filtered_metric_data = filtered_df.dropna(subset=['Hours per day', metric])

# Create a scatter plot faceted by genre
fig1 = px.scatter(
    filtered_metric_data,
    x="Hours per day",
    y=metric,
    color="Fav genre",
    facet_col="Fav genre",  # Separate the plot by genre
    facet_col_wrap=4,  # Adjust the number of plots per row
    title=f"Hours per Day vs. {metric} (Faceted by Genre)",
    labels={"Hours per day": "Hours of Music Listening per Day"},
)

# Update the layout to make it more readable
fig1.update_layout(
    showlegend=False,
    height=600,
    title_x=0.5,
)

st.plotly_chart(fig1, use_container_width=True)


# SECTION 2: Boxplot - Mental Health by Genre
st.subheader("🎵 Mental Health Levels by Favorite Genre")

fig2 = px.box(
    filtered_df,
    x="Fav genre",
    y=metric,
    points="all",
    color="Fav genre",
    title=f"{metric} Levels Across Favorite Genres"
)
st.plotly_chart(fig2, use_container_width=True)

# SECTION 3: Correlation heatmap
st.subheader("Correlation Between Mental Health and Listening Habits")

# Mapping for frequencies
freq_mapping = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Very frequently": 3
}

# Identify columns
genre_freq_cols = [col for col in df.columns if col.startswith("Frequency [")]
mental_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

# Ensure all required columns are in the filtered DataFrame
available_cols = [col for col in genre_freq_cols + mental_cols if col in filtered_df.columns]

# Check if enough data to compute correlation
if len(filtered_df) >= 5 and all(col in filtered_df.columns for col in mental_cols):
    corr_data = filtered_df[available_cols].copy()

    # Convert frequencies
    for col in genre_freq_cols:
        if col in corr_data.columns:
            corr_data[col] = corr_data[col].map(freq_mapping)

    # Drop NaNs
    corr_data = corr_data.dropna()

    # Clean column names
    corr_data.columns = [col.replace("Frequency [", "").replace("]", "") for col in corr_data.columns]

    # Compute and plot correlation matrix
    corr_matrix = corr_data.corr().loc[mental_cols, corr_data.columns.difference(mental_cols)]

    fig3, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

else:
    st.warning("Not enough data in the current selection to compute correlation.")
