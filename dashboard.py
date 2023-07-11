# Import libraries
import nltk
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import gensim
from gensim import corpora
from pprint import pprint
from collections import Counter
from matplotlib import colors as mcolors
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# Read the data from the CSV file
df = pd.read_csv('user_reviews.csv')

#get the stopword and punctuation list from nltk library
stop_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)

#remove the negate word from the stopwords list
negate_words = ['no', 'not', 'none', 'neither', 'never', 'nothing', 'nowhere', 'nor']

filtered_stop_words = []

for word in stop_words:
  if word not in negate_words:
    filtered_stop_words.append(word)

for word in stop_words:
  if "n't" in word:
    filtered_stop_words.remove(word)

#define a function to remove the stopwords and combine the remaining words as the sentence
def remove_stopwords(review):
    tokens = nltk.tokenize.word_tokenize(review)
    filtered_review = [token for token in tokens if not token.lower() in filtered_stop_words]
    return ' '.join(filtered_review)


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenization and removing stopwords
    text = remove_stopwords(text)
    
    return text

def main():
    # Title of the app
    st.title("Restaurant User Review Sentiment Analysis Dashboard")

    # Refresh the dashboard every 10 seconds
    st_autorefresh(interval=10 * 1000)

    # Filter by sentiment
    sentiment_filter = st.selectbox("Filter by Sentiment", options=['All', 'Positive', 'Neutral', 'Negative'])

    # Filtered DataFrame based on sentiment selection
    if sentiment_filter == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['Sentiment'] == sentiment_filter]

    # Filter by sarcasm
    sarcasm_filter = st.sidebar.checkbox("Show Sarcasm")
    if sarcasm_filter:
        filtered_df = filtered_df[filtered_df['Sarcasm'] == 'Sarcasm']

    # Display the filtered DataFrame on the left panel
    st.sidebar.subheader("Filtered Data")
    st.sidebar.write(filtered_df)

    # Scorecard
    st.subheader("Scorecard")
    total_reviews = len(filtered_df)
    positive_reviews = len(filtered_df[filtered_df['Sentiment'] == 'Positive'])
    neutral_reviews = len(filtered_df[filtered_df['Sentiment'] == 'Neutral'])
    negative_reviews = len(filtered_df[filtered_df['Sentiment'] == 'Negative'])
    average_rating = filtered_df['Rating'].mean()
    
    # Style the scorecard as a rectangle
    st.markdown(
        """
        <style>
        .scorecard {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .scorecard-item {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the scorecard
    st.markdown(
        f"""
        <div class="scorecard">
            <div class="scorecard-item">
                <h3>Total Reviews</h3>
                <p>{total_reviews}</p>
            </div>
            <div class="scorecard-item">
                <h3>Average Rating</h3>
                <p>{round(average_rating, 2)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create a two-column layout
    col1, col2, col3 = st.columns(3)

    # Pie Chart - Sarcasm Class Distribution
    with col1:
        st.subheader('Sentiment Distribution')
        sentiment_counts = filtered_df['Sentiment'].value_counts()
        sentiment_labels = sentiment_counts.index
        sentiment_values = sentiment_counts.values
        colors = ['green', 'blue', 'red']  # Define the colors for each bar
        fig1, ax1 = plt.subplots()
        ax1.bar(sentiment_labels, sentiment_values, color=colors)
        ax1.set_xlabel("Sentiment")
        ax1.set_ylabel("Count")
        ax1.set_title("")
        st.pyplot(fig1)

    # Bar Chart - Sentiment Distribution
    with col2:
        st.subheader('Sarcasm Class Distribution')
        sarcasm_counts = filtered_df['Sarcasm'].value_counts()
        sarcasm_labels = sarcasm_counts.index
        sarcasm_values = sarcasm_counts.values
        fig2, ax2 = plt.subplots()
        ax2.pie(sarcasm_values, labels=sarcasm_labels, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        plt.title('')
        st.pyplot(fig2)

    # Calculate the number of reviews per day for each sentiment
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    filtered_df['Day'] = filtered_df['Date'].dt.date
    reviews_per_day = filtered_df.groupby(['Sentiment', 'Day']).size().unstack(level=0).fillna(0)
    
    # Line chart - Number of reviews over days by sentiment
    with col3:
        st.subheader('Number of Reviews Over Days by Sentiment')
        fig, ax = plt.subplots()
        for sentiment in reviews_per_day.columns:
            ax.plot(reviews_per_day.index, reviews_per_day[sentiment], marker='o', label=sentiment)
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Reviews')
        ax.set_title('')
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    # Rating vs Sentiment scatter plot
    st.subheader('Rating vs Sentiment')
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=filtered_df, x='Rating', y='Sentiment', hue='Sentiment', palette='Set1')
    ax3.set_xlabel('Rating')
    ax3.set_ylabel('Sentiment')
    ax3.set_title('')
    st.pyplot(fig3)

    # Word Cloud
    st.subheader('Word Cloud')
    text = ' '.join(filtered_df['Review'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig4, ax4 = plt.subplots()
    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.axis('off')
    ax4.set_title('')
    st.pyplot(fig4)
    
    # Calculate the top words for each sentiment
    top_words = {'Positive': Counter(), 'Neutral': Counter(), 'Negative': Counter()}

    for review in filtered_df['Review']:
        sentiment = filtered_df.loc[filtered_df['Review'] == review, 'Sentiment'].iloc[0]
        processed_text = preprocess_text(review)
        top_words[sentiment].update(processed_text.split())

    # Create bar chart for top positive, neutral, and negative words
    st.subheader("Top 10 Positive, Neutral, and Negative Words")
    fig, ax = plt.subplots()
    x_pos = range(10)
    bar_width = 0.25

    for i, (sentiment, word_counts) in enumerate(top_words.items()):
        top_10_words = dict(word_counts.most_common(10))
        ax.bar([x + i * bar_width for x in x_pos], top_10_words.values(), bar_width, label=sentiment)

    ax.set_xticks([x + bar_width for x in x_pos])
    ax.set_xticklabels(top_10_words.keys(), rotation=45)
    ax.set_xlabel("Words")
    ax.set_ylabel("Count")
    ax.set_title("Top 10 Positive, Neutral, and Negative Words")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
