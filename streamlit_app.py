import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend to avoid GUI issues when running in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

# =====================
# Page config
# =====================

st.set_page_config(
    page_title="Netflix Content Explorer",
    page_icon="🎬",
    layout="wide"
)

# =====================
# 1. Load & cache data
# =====================

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")

    # Keep original date for debugging
    df['date_added_orig'] = df['date_added']

    # Parse date_added
    df['date_added'] = pd.to_datetime(
        df['date_added'].str.strip(),
        errors='coerce'
    )
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month

    # Handle missing text fields
    df['listed_in'] = df['listed_in'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')

    # Genres exploded
    df_genres = df.assign(
        genre=df['listed_in'].str.split(', ')
    ).explode('genre')

    # Duration
    df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['duration_type'] = df['duration'].str.extract(r'([A-Za-z]+)')

    # Extra flags
    df['is_movie'] = (df['type'] == 'Movie').astype(int)
    kids_ratings = ['TV-Y', 'TV-Y7', 'TV-G', 'G', 'PG', 'TV-PG']
    df['is_kids'] = df['rating'].isin(kids_ratings).astype(int)

    # Countries exploded + region
    region_map = {
        'United States': 'North America',
        'India': 'Asia',
        'United Kingdom': 'Europe',
        'Japan': 'Asia',
        'Canada': 'North America',
        'France': 'Europe',
        'Germany': 'Europe',
        'Spain': 'Europe',
        'South Korea': 'Asia',
        'Brazil': 'South America'
    }

    df_countries = df.assign(
        country=df['country'].str.split(', ')
    ).explode('country')
    df_countries['region'] = df_countries['country'].map(region_map).fillna('Other')

    return df, df_genres, df_countries


# Load data once
df, df_genres, df_countries = load_data()

# =====================
# 2. Sidebar controls
# =====================

st.sidebar.title("Filters")

years = sorted(df['year_added'].dropna().unique())
min_year, max_year = (min(years), max(years)) if years else (None, None)

year_range = st.sidebar.slider(
    "Year Added Range",
    min_value=int(min_year) if min_year else 2008,
    max_value=int(max_year) if max_year else 2021,
    value=(
        int(min_year) if min_year else 2015,
        int(max_year) if max_year else 2021
    )
)

content_type = st.sidebar.multiselect(
    "Content Type",
    options=sorted(df['type'].dropna().unique()),
    default=list(sorted(df['type'].dropna().unique()))
)

selected_region = st.sidebar.multiselect(
    "Region (from exploded countries)",
    options=sorted(df_countries['region'].dropna().unique()),
    default=list(sorted(df_countries['region'].dropna().unique()))
)

# Filtered base DataFrame
df_filtered = df[
    (df['year_added'].between(year_range[0], year_range[1])) &
    (df['type'].isin(content_type))
]

df_countries_filtered = df_countries[
    (df_countries['year_added'].between(year_range[0], year_range[1])) &
    (df_countries['region'].isin(selected_region))
]

df_genres_filtered = df_genres[
    (df_genres['year_added'].between(year_range[0], year_range[1])) &
    (df_genres['type'].isin(content_type))
]

# =====================
# 3. Main page layout
# =====================

st.title("Netflix Content Explorer")
st.caption("Interactive EDA of the Netflix Titles dataset.")

st.markdown(
    "This dashboard lets you explore how Netflix's catalog evolves over time "
    "by **type**, **genre**, **rating**, **country**, and **region**."
)

st.markdown("### Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Titles (filtered)", len(df_filtered))
with col2:
    st.metric("Movies", int((df_filtered['type'] == 'Movie').sum()))
with col3:
    st.metric("TV Shows", int((df_filtered['type'] == 'TV Show').sum()))

st.markdown("### Data Snapshot")
st.dataframe(
    df_filtered[
        [
            'title', 'type', 'country', 'date_added', 'release_year',
            'rating', 'duration', 'listed_in'
        ]
    ].head(20)
)

# =====================
# 4. Plots with Streamlit
# =====================

# Helper: show Matplotlib/Seaborn fig in Streamlit
def show_fig():
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()

# 4.1 Count of Movies vs TV Shows
st.markdown("### Movies vs TV Shows (Filtered)")

plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df_filtered, x='type', palette="Set2")
plt.title("Count of Movies vs TV Shows", fontsize=14)
plt.xlabel("")
plt.ylabel("Number of Titles")

for container in ax.containers:
    ax.bar_label(container, fontsize=10)

show_fig()

# 4.2 Titles added per year
st.markdown("### Titles Added per Year")

titles_per_year = df_filtered['year_added'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
ax = titles_per_year.plot(kind='bar', color="#4C72B0")
plt.title("Number of Titles Added per Year", fontsize=14)
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
show_fig()

# 4.3 Movies vs TV Shows added over time
st.markdown("### Movies vs TV Shows Added Over Time")

year_type = (
    df_filtered.groupby(['year_added', 'type'])['show_id']
    .count()
    .reset_index()
    .dropna(subset=['year_added'])
)

plt.figure(figsize=(10, 5))
sns.lineplot(
    data=year_type, x='year_added', y='show_id',
    hue='type', marker='o'
)
plt.title("Movies vs TV Shows Added Over Time", fontsize=14)
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
show_fig()

# 4.4 Heatmap of ratings by year
st.markdown("### Ratings by Year (Heatmap)")

rating_year = df_filtered.pivot_table(
    index='rating',
    columns='year_added',
    values='show_id',
    aggfunc='count'
).fillna(0)

plt.figure(figsize=(12, 6))
sns.heatmap(rating_year, cmap="Reds")
plt.title("Number of Titles by Rating and Year Added", fontsize=14)
plt.xlabel("Year Added")
plt.ylabel("Rating")
show_fig()

# 4.5 Top genres
st.markdown("### Top Genres (Filtered)")

top_genres = df_genres_filtered['genre'].value_counts().head(15)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, color="#55A868")
plt.title("Top 15 Genres", fontsize=14)
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
show_fig()

# 4.6 Top genres by type
st.markdown("### Top Genres by Type")

df_genres_type = df_filtered.assign(
    genre=df_filtered['listed_in'].str.split(', ')
).explode('genre')
top_genres10 = df_genres_type['genre'].value_counts().index[:10]
genre_type_counts = (
    df_genres_type[df_genres_type['genre'].isin(top_genres10)]
    .groupby(['genre', 'type'])
    .size()
    .reset_index(name='count')
)

plt.figure(figsize=(8, 6))
sns.barplot(
    data=genre_type_counts, x='count', y='genre',
    hue='type', palette="Set2"
)
plt.title("Top Genres by Type", fontsize=14)
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
show_fig()

# 4.7 Top countries and regions
st.markdown("### Top Countries (Filtered by Region)")

top_countries = df_countries_filtered['country'].value_counts().head(10)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, color="#8172B2")
plt.title("Top 10 Content-Producing Countries", fontsize=14)
plt.xlabel("Number of Titles")
plt.ylabel("Country")
show_fig()

st.markdown("### Titles by Region")

region_counts = df_countries_filtered['region'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=region_counts.index, y=region_counts.values, palette="Set3")
plt.title("Titles by Region", fontsize=14)
plt.xlabel("Region")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
show_fig()

st.markdown("#### Notes")
st.write(
    "Use the sidebar to filter by **year range**, **content type**, and **region** "
    "to see how the Netflix catalog changes under different conditions."
)
