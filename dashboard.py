# Import libraries
import nltk
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
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
import joblib
from datetime import datetime


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Set wide mode layout
st.set_page_config(layout="wide")

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
    
    # Description of the app
    st.write("This dashboard is designed for the F&B industry and uses a model trained on user reviews from Malaysian restaurants collected from Google Reviews. The model simultaneously predicts sentiment and sarcasm in the reviews. Users must submit their review by accessing the 'Submit Review' page situated on the left panel. This dashboard page is intended for executive use and is password-protected. This platform allows executives to gain valuable insights into user sentiments and sarcasm in the F&B industry, which can help them make informed decisions.")
    
    st.write("The goal of this dashboard is to raise awareness of the importance of data-driven decision-making processes for SMEs and to train them on how to use real-time dashboards.")

    st.write("To access the data and view the dashboard, please enter the password 'admin123'.")
    
    # Add a divider under the title
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Refresh the dashboard every 10 seconds
    st_autorefresh(interval=10 * 1000)
    
    # Passcode input
    passcode = st.text_input("Please enter the passcode:", type="password")

    if passcode == "admin123":
        display_dashboard()
    elif passcode != "":
        st.warning("Incorrect passcode. Please try again.")
    
def display_dashboard():

    # Filter by sentiment
    sentiment_filter = st.sidebar.selectbox("Filter by Sentiment", options=['All', 'Positive', 'Neutral', 'Negative'])

    # Filtered DataFrame based on sentiment selection
    if sentiment_filter == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['Sentiment'] == sentiment_filter]

    # Filter by sarcasm
    sarcasm_filter = st.sidebar.checkbox("Show Sarcasm")
    if sarcasm_filter:
        filtered_df = filtered_df[filtered_df['Sarcasm'] == 'Sarcasm']

    # Display the filtered DataFrame on the main panel
    with st.expander("View User Records", expanded=False):
        st.write(filtered_df)

    # Scorecard
    st.subheader("Scorecard")
    total_reviews = len(filtered_df)
    positive_reviews = len(filtered_df[filtered_df['Sentiment'] == 'Positive'])
    neutral_reviews = len(filtered_df[filtered_df['Sentiment'] == 'Neutral'])
    negative_reviews = len(filtered_df[filtered_df['Sentiment'] == 'Negative'])
    sarcasm_count = len(filtered_df[filtered_df['Sarcasm'] == 'Sarcasm'])
    average_rating = filtered_df['Rating'].mean()
    

    # Style the scorecards with modern design and hover effect
    st.markdown(
        """
        <style>
        .scorecards {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .scorecard-item {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
            width: 200px;
            margin-right: 10px;
            transition: transform 0.2s ease-in-out;
        }
        .scorecard-item:hover {
            transform: scale(1.02);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .scorecard-item h3 {
            font-size: 16px;
            margin-bottom: 5px;
        }
        .scorecard-item p {
            font-size: 20px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the scorecards
    st.markdown(
        f"""
        <div class="scorecards">
            <div class="scorecard-item">
                <h3>Total Reviews</h3>
                <p>{total_reviews}</p>
            </div>
            <div class="scorecard-item">
                <h3>Average Rating</h3>
                <p>{round(average_rating, 2)}</p>
            </div>
            <div class="scorecard-item">
                <h3>Positive Reviews</h3>
                <p>{positive_reviews}</p>
            </div>
            <div class="scorecard-item">
                <h3>Neutral Reviews</h3>
                <p>{neutral_reviews}</p>
            </div>
            <div class="scorecard-item">
                <h3>Negative Reviews</h3>
                <p>{negative_reviews}</p>
            </div>
            <div class="scorecard-item">
                <h3>Sarcasm Count</h3>
                <p>{sarcasm_count}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )




    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Bar Chart - Sentiment Class Distribution
    with col1:
        st.subheader('Sentiment Distribution')
        sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        sentiment_chart = alt.Chart(sentiment_counts).mark_bar().encode(
            x='Sentiment',
            y='Count',
            color='Sentiment',
            tooltip=['Sentiment', 'Count']
        ).properties(
            width=600,  # Set the width of the chart
            height=400  # Set the height of the chart
        )
        st.altair_chart(sentiment_chart)

    # Donut Chart - Sarcasm Class Distribution
    with col2:
        sarcasm_counts = filtered_df['Sarcasm'].value_counts()
        sarcasm_data = pd.DataFrame({'Sarcasm': sarcasm_counts.index, 'Count': sarcasm_counts.values})

        fig = px.pie(sarcasm_data, values='Count', names='Sarcasm', hole=0.5,
                     labels={'Count': 'Count', 'Sarcasm': 'Sarcasm'},
                     title='Sarcasm Class Distribution')

        fig.update_traces(textposition='inside', textinfo='percent+label')

        fig.update_layout(width=600, height=400)

        st.plotly_chart(fig)
    
    # Bar Chart - Rating Distribution
    st.subheader('Rating Distribution')
    rating_counts = filtered_df['Rating'].value_counts().reset_index()
    rating_counts.columns = ['Rating', 'Count']

    # Calculate the mean rating
    mean_rating = filtered_df['Rating'].mean()

    fig = px.bar(rating_counts, x='Rating', y='Count', color='Rating', 
                 labels={'Rating': 'Rating', 'Count': 'Count'})

    fig.add_hline(y=mean_rating, line_dash='dash', line_color='red', 
                  annotation_text=f'Mean Rating: {round(mean_rating, 2)}',
                  annotation_position='bottom right')

    fig.update_layout(width=1200, height=400)
    st.plotly_chart(fig)
    

    # Create a two-column layout
    col3, col4 = st.columns(2)    

    
    # Calculate the number of reviews per day for each sentiment
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    filtered_df['Day'] = filtered_df['Date'].dt.date
    reviews_per_day = filtered_df.groupby(['Sentiment', 'Day']).size().unstack(level=0).fillna(0)

    with col3:
        # Line chart - Number of reviews over days by sentiment
        st.subheader('Number of Reviews Over Days by Sentiment')

        fig = px.line(reviews_per_day, x=reviews_per_day.index, y=reviews_per_day.columns,
                      labels={'x': 'Date', 'y': 'Number of Reviews'}, markers=True)

        fig.update_layout(title='', xaxis_title='Date', yaxis_title='Number of Reviews', legend_title='Sentiment')
        fig.update_xaxes(tickangle=45, tickformat="%Y-%m-%d")

        st.plotly_chart(fig)
    
    with col4:
        
        # Violin chart - Rating Distribution by Sentiments
        st.subheader('Rating Distribution by Sentiments')
        # Create the violin plot using Plotly Express
        fig = px.violin(filtered_df, x='Rating', y='Sentiment', box=True, points='all')

        # Customize the plot layout
        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Sentiment",
            width=600,
            height=500
        )

        # Display the plot within the Streamlit app
        st.plotly_chart(fig)

    # Word Cloud
    st.subheader('Word Cloud')
    text = ' '.join(filtered_df['Review'])

    # Get the top 20 words

    # Preprocess text
    processed_text = preprocess_text(text)
    word_counts = Counter(processed_text.split())
    top_words = dict(word_counts.most_common(50))

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(top_words)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Top 20 Words')
    st.pyplot(fig)
    
    # Calculate the top words for each sentiment
    top_words = {'Positive': Counter(), 'Neutral': Counter(), 'Negative': Counter()}

    for review in df['Review']:
        sentiment = df.loc[df['Review'] == review, 'Sentiment'].iloc[0]
        processed_text = preprocess_text(review)
        top_words[sentiment].update(processed_text.split())

    # Get the top 10 words for each sentiment
    top_positive_words = dict(top_words['Positive'].most_common(10))
    top_neutral_words = dict(top_words['Neutral'].most_common(10))
    top_negative_words = dict(top_words['Negative'].most_common(10))

    # Create data for the bar chart
    data = [
        go.Bar(name='Positive', x=list(top_positive_words.keys()), y=list(top_positive_words.values())),
        go.Bar(name='Neutral', x=list(top_neutral_words.keys()), y=list(top_neutral_words.values())),
        go.Bar(name='Negative', x=list(top_negative_words.keys()), y=list(top_negative_words.values()))
    ]

    # Set the layout for the bar chart
    layout = go.Layout(
        title='Top 10 Positive, Neutral, and Negative Words',
        xaxis=dict(title='Words'),
        yaxis=dict(title='Count'),
        barmode='group',
        width=1200,
        height=500
    )

    # Create the figure with data and layout
    fig = go.Figure(data=data, layout=layout)

    # Display the bar chart using Plotly
    st.plotly_chart(fig)

    
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
