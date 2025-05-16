import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Music & Mental Health", layout="wide")
st.title("How music affects mental health?")

st.markdown("""
This interactive dashboard explores how music listening habits may relate to mental health conditions,  
based on data from **Music & Mental Health Survey Results**

You can filter the dataset by age range and favorite genres using the sidebar  

The visualizations cover correlations between music engagement and symptoms of **anxiety**, **depression**, **insomnia**, and **OCD**

Dataset source: [Kaggle - Music & Mental Health Survey Results](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)


""")

# data prep
df = pd.read_csv("mxmh_survey_results.csv")
df.columns = [col.strip() for col in df.columns]
num_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Hours per day', 'BPM']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Hours per day', 'Anxiety', 'Depression', 'Insomnia', 'OCD'])

# sidebar
st.sidebar.header("Filter Data")
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (18, 65))
genres = st.sidebar.multiselect("Favorite Genres", df["Fav genre"].dropna().unique(), default=df["Fav genre"].dropna().unique())

filtered_df = df[(df["Age"] >= age_range[0]) &
                 (df["Age"] <= age_range[1]) &
                 (df["Fav genre"].isin(genres))]

# section 0: basic statistics
st.subheader("Basic Participant Information")
st.markdown("""
This section provides an overview of the survey participants based on your selected filters in the sidebar  
It includes distributions of participants by age and by their favorite music genres
""")
col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Number of Participants by Age")
    age_counts = filtered_df["Age"].value_counts().sort_index().reset_index()
    age_counts.columns = ["Age", "Count"]
    fig_age_bar = px.bar(age_counts, x="Age", y="Count",
                         title="Number of Participants by Age",
                         labels={"Age": "Age", "Count": "Number of Participants"},
                         color_discrete_sequence=["#00CC96"])
    fig_age_bar.update_layout(title_x=0.5)
    st.plotly_chart(fig_age_bar, use_container_width=True)

with col2:
    st.subheader("🎶 Number of Participants by Favorite Genre")
    genre_counts = filtered_df["Fav genre"].value_counts().reset_index()
    genre_counts.columns = ["Fav genre", "Count"]
    fig_genre_bar = px.bar(genre_counts, x="Fav genre", y="Count",
                           title="Number of Participants by Favorite Genre",
                           labels={"Fav genre": "Favorite Genre", "Count": "Number of Participants"},
                           color="Fav genre")
    fig_genre_bar.update_layout(title_x=0.5, showlegend=False)
    st.plotly_chart(fig_genre_bar, use_container_width=True)


# section 1: mental health score by age

st.subheader("📈 Average Mental Health Score by Age")
st.markdown("Line plot depicting average scores of selected mental health metrics by age, highlighting trends across age groups")
metric_line = st.selectbox("Select a mental health metric for age trend", ['Anxiety', 'Depression', 'Insomnia', 'OCD'], key="line_metric")

avg_by_age = filtered_df.groupby("Age")[metric_line].mean().reset_index()

fig_line = px.line(avg_by_age, x="Age", y=metric_line, markers=True,
                   title=f"Average {metric_line} Score by Age",
                   labels={"Age": "Age", metric_line: f"Avg {metric_line}"})

fig_line.update_layout(title_x=0.5)
st.plotly_chart(fig_line, use_container_width=True)


# section 2: box plots - mental health by genre

st.subheader("🎵 Mental Health Levels by Favorite Genre")
st.markdown("Box plots showing distribution and variability of selected mental health scores across different favorite music genres")
metric_box = st.selectbox("Select a mental health metric to visualize by genre",
                          ['Anxiety', 'Depression', 'Insomnia', 'OCD'],
                          key="boxplot_metric")

fig2 = px.box(
    filtered_df,
    x="Fav genre",
    y=metric_box,
    points="all",
    color="Fav genre",
    title=f"{metric_box} Levels Across Favorite Genres",
    labels={"Fav genre": "Favorite Genre", metric_box: metric_box}
)
fig2.update_layout(title_x=0.5, showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

# section 3: correlation heatmap

st.subheader("🎵 Correlation Between Mental Health and Listening Habits")
st.markdown("Heatmap illustrating correlations between frequency of listening to various genres and mental health symptoms")

freq_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Very frequently": 3}
genre_freq_cols = [col for col in df.columns if col.startswith("Frequency [")]
mental_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

available_cols = [col for col in genre_freq_cols + mental_cols if col in filtered_df.columns]

if len(filtered_df) >= 5 and all(col in filtered_df.columns for col in mental_cols):
    corr_data = filtered_df[available_cols].copy()

    for col in genre_freq_cols:
        if col in corr_data.columns:
            corr_data[col] = corr_data[col].map(freq_mapping)

    corr_data = corr_data.dropna()
    corr_data.columns = [col.replace("Frequency [", "").replace("]", "") for col in corr_data.columns]
    corr_matrix = corr_data.corr().loc[mental_cols, corr_data.columns.difference(mental_cols)]

    fig3, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)
else:
    st.warning("Not enough data in the current selection to compute correlation")

# section 4: average mental health by listening time category

st.subheader("🎵 Average Mental Health by Listening Time Category")
st.markdown("""
Line plot showing average selected mental health metric by categorized daily music listening time
""")

metric = st.selectbox("Select a mental health metric for listening time trend",
                      ['Anxiety', 'Depression', 'Insomnia', 'OCD'], key="listening_time_metric")

filtered_metric_data = filtered_df.dropna(subset=['Hours per day', metric])

max_hours = filtered_metric_data['Hours per day'].max()
max_ceil = math.ceil(max_hours)

def categorize_hours_dynamic(h):
    for i in range(max_ceil):
        if h < i + 1:
            return f"{i}-{i+1} hours" if i > 0 else "< 1 hour"
    return f"{max_ceil}+ hours"

filtered_metric_data['Listening Time Category'] = filtered_metric_data['Hours per day'].apply(categorize_hours_dynamic)

avg_metric_by_time = filtered_metric_data.groupby('Listening Time Category')[metric].mean().reset_index()

fig_line = px.line(
    avg_metric_by_time,
    x='Listening Time Category',
    y=metric,
    title=f"Average {metric} Score by Listening Time Category",
    labels={'Listening Time Category': 'Daily Listening Time', metric: f'Average {metric}'},
    markers=True
)
fig_line.update_layout(title_x=0.5, xaxis_tickangle=-45)
st.plotly_chart(fig_line, use_container_width=True)
