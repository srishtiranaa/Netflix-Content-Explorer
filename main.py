import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# =====================
# 1. Load data
# =====================
df = pd.read_csv("netflix_titles.csv")

# Quick info checks
print(df.head())
print(df.info())
print(df.describe(include="all"))

print("Missing values before cleaning:")
print(df.isna().sum())

# =====================
# 2. Basic cleaning
# =====================

# Keep original date column for debugging
df['date_added_orig'] = df['date_added']

# Strip whitespace and parse date_added; coerce invalid formats to NaT
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(),
                                  errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Handle missing listed_in and country before splitting
df['listed_in'] = df['listed_in'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

# Explode genres
df_genres = df.assign(genre=df['listed_in'].str.split(', ')).explode('genre')

# Duration: extract numeric part and type (min / Season(s))
df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(float)
df['duration_type'] = df['duration'].str.extract(r'([A-Za-z]+)')

# Extra flags (optional but useful)
df['is_movie'] = (df['type'] == 'Movie').astype(int)

kids_ratings = ['TV-Y', 'TV-Y7', 'TV-G', 'G', 'PG', 'TV-PG']
df['is_kids'] = df['rating'].isin(kids_ratings).astype(int)

# Explode countries with simple region mapping for top countries
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

df_countries = df.assign(country=df['country'].str.split(', ')).explode('country')
df_countries['region'] = df_countries['country'].map(region_map).fillna('Other')

print("Missing values after cleaning:")
print(df.isna().sum())

# =====================
# Helper for saving plots
# =====================

def save_current_fig(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# =====================
# 3. Core EDA plots
# =====================

# 3.1 Count of Movies vs TV Shows
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='type', palette="Set2")
plt.title("Count of Movies vs TV Shows", fontsize=14)
plt.xlabel("")
plt.ylabel("Number of Titles")

# Add counts on bars
for container in ax.containers:
    ax.bar_label(container, fontsize=10)

save_current_fig("01_count_movies_vs_tv.png")

# 3.2 Titles added per year
titles_per_year = df['year_added'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
ax = titles_per_year.plot(kind='bar', color="#4C72B0")
plt.title("Number of Titles Added per Year", fontsize=14)
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
save_current_fig("02_titles_per_year.png")

# 3.3 Movies vs TV Shows added over time (lineplot)
year_type = (
    df.groupby(['year_added', 'type'])['show_id']
      .count()
      .reset_index()
      .dropna(subset=['year_added'])
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=year_type, x='year_added', y='show_id',
             hue='type', marker='o')
plt.title("Movies vs TV Shows Added Over Time", fontsize=14)
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
save_current_fig("03_movies_vs_tv_over_time.png")

# =====================
# 4. Advanced time-based visuals
# =====================

# 4.1 Stacked share of Movies vs TV Shows over time
year_pivot = (
    year_type.pivot(index='year_added', columns='type', values='show_id')
             .fillna(0)
)
year_pivot_pct = year_pivot.div(year_pivot.sum(axis=1), axis=0)

plt.figure(figsize=(10, 5))
year_pivot_pct.plot(kind='bar', stacked=True, colormap='Set2', ax=plt.gca())
plt.title("Share of Movies vs TV Shows Added Over Time", fontsize=14)
plt.xlabel("Year Added")
plt.ylabel("Share of Titles")
plt.legend(title="Type")
save_current_fig("04_share_movies_vs_tv_over_time.png")

# 4.2 Heatmap of ratings by year_added
rating_year = df.pivot_table(
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
save_current_fig("05_heatmap_rating_year.png")

# 4.3 Genre trend over time for selected genres
focus_genres = ['Documentaries', 'International Movies', 'Dramas']
genres_year = (
    df_genres[df_genres['genre'].isin(focus_genres)]
    .groupby(['year_added', 'genre'])['show_id']
    .count()
    .reset_index()
    .dropna(subset=['year_added'])
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=genres_year, x='year_added', y='show_id',
             hue='genre', marker='o')
plt.title("Trend of Selected Genres Over Time", fontsize=14)
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
save_current_fig("06_genre_trends.png")

# =====================
# 5. Genres, ratings, countries
# =====================

# 5.1 Top 15 genres overall
top_genres = df_genres['genre'].value_counts().head(15)

plt.figure(figsize=(8, 6))
ax = sns.barplot(x=top_genres.values, y=top_genres.index, color="#55A868")
plt.title("Top 15 Genres on Netflix", fontsize=14)
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
save_current_fig("07_top_15_genres.png")

# 5.2 Top genres by type (Movies vs TV Shows)
df_genres_type = df.assign(genre=df['listed_in'].str.split(', ')).explode('genre')
top_genres10 = df_genres_type['genre'].value_counts().index[:10]
genre_type_counts = (
    df_genres_type[df_genres_type['genre'].isin(top_genres10)]
    .groupby(['genre', 'type'])
    .size()
    .reset_index(name='count')
)

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=genre_type_counts, x='count', y='genre',
                 hue='type', palette="Set2")
plt.title("Top Genres by Type", fontsize=14)
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
save_current_fig("08_top_genres_by_type.png")

# 5.3 Distribution of content ratings
rating_counts = df['rating'].value_counts()

plt.figure(figsize=(8, 6))
ax = sns.barplot(x=rating_counts.values, y=rating_counts.index, color="#C44E52")
plt.title("Distribution of Content Ratings", fontsize=14)
plt.xlabel("Number of Titles")
plt.ylabel("Rating")
save_current_fig("09_rating_distribution.png")

# 5.4 Top 10 content-producing countries
top_countries = df_countries['country'].value_counts().head(10)

plt.figure(figsize=(8, 6))
ax = sns.barplot(x=top_countries.values, y=top_countries.index, color="#8172B2")
plt.title("Top 10 Content-Producing Countries", fontsize=14)
plt.xlabel("Number of Titles")
plt.ylabel("Country")
save_current_fig("10_top_10_countries.png")

# 5.5 Titles by region (optional extra)
region_counts = df_countries['region'].value_counts()

plt.figure(figsize=(6, 4))
ax = sns.barplot(x=region_counts.index, y=region_counts.values, palette="Set3")
plt.title("Titles by Region", fontsize=14)
plt.xlabel("Region")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
save_current_fig("11_titles_by_region.png")

print("All plots saved to PNG files in the current directory.")
