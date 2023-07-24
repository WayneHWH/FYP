# Import libraries
import nltk
from nltk import ngrams
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
from nltk import bigrams
from matplotlib import colors as mcolors
import string
import joblib
from datetime import datetime
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


nltk.download('wordnet')
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
    st.write("This dashboard is designed for the F&B industry and uses a model trained on user reviews from Malaysian restaurants collected from Google Reviews. The model simultaneously predicts sentiment and sarcasm in the reviews. This platform allows executives to gain valuable insights into user sentiments in the F&B industry, which can help them make informed decisions.")
    
    st.write("The goal of this dashboard is to raise awareness of the importance of data-driven decision-making processes for SMEs and to train them on how to use real-time dashboards.")
    
    st.write("1. Users must submit their review by accessing the 'Submit Review' page situated on the left panel.")
    
    st.write("2. This dashboard page is intended for executive use and is password-protected.")
    
    st.write("3. To access the data and view the dashboard, please enter the password 'admin123'")
    
    
    # Add a divider under the title
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Refresh the dashboard every 120 seconds
    st_autorefresh(interval=120 * 1000)
    
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

    # Date filter on the left panel
    st.sidebar.subheader("Date Filter")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")

    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime type

        if start_date and end_date:
            filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        else:
            filtered_df = df
    else:
        filtered_df = pd.DataFrame()

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
        
        # Violin chart - Rating Distribution by Sentiments or Sarcasm
        st.subheader('Rating Distribution')
        chart_type = st.selectbox("Select Chart Type", options=["Sentiment", "Sarcasm"])

        if chart_type == "Sentiment":
            chart_data = filtered_df
            y_label = "Sentiment"
        elif chart_type == "Sarcasm":
            chart_data = filtered_df
            y_label = "Sarcasm"

        # Create the violin plot using Plotly Express
        fig = px.violin(chart_data, x='Rating', y=y_label, box=True, points='all')

        # Customize the plot layout
        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title=y_label,
            width=600,
            height=500
        )

        # Display the plot within the Streamlit app
        st.plotly_chart(fig)
        
        
        
        
    
    ############ TOP 10 POSITIVE, NEUTRAL AND NEGATIVE WORD COUNT BAR CHART ############  
    
    # Calculate the top words and bigrams for each sentiment and sarcasm label
    top_words = {
        'Positive': Counter(),
        'Neutral': Counter(),
        'Negative': Counter(),
        'Sarcasm': Counter()
    }
    top_bigrams = {
        'Positive': Counter(),
        'Neutral': Counter(),
        'Negative': Counter(),
        'Sarcasm': Counter()
    }

    for review, sarcasm in zip(df['Review'], df['Sarcasm']):
        sentiment = df.loc[df['Review'] == review, 'Sentiment'].iloc[0]
        processed_text = preprocess_text(review)
        tokens = processed_text.split()
        top_words[sentiment].update(tokens)
        if sarcasm == 'Sarcasm':
            top_words['Sarcasm'].update(tokens)
        bigrams = list(ngrams(tokens, 2))  # Generate bigrams
        top_bigrams[sentiment].update(bigrams)
        if sarcasm == 'Sarcasm':
            top_bigrams['Sarcasm'].update(bigrams)

    # Get the selected option from the select box
    selected_option = st.selectbox("Select Chart", options=['Sentiment', 'Sarcasm'])

    # Determine the top words/bigrams and title based on the selected option
    if selected_option == 'Sentiment':
        top_words_dict = {
            'Positive': dict(top_words['Positive'].most_common(10)),
            'Neutral': dict(top_words['Neutral'].most_common(10)),
            'Negative': dict(top_words['Negative'].most_common(10))
        }
        top_bigrams_dict = {
            'Positive': dict(top_bigrams['Positive'].most_common(10)),
            'Neutral': dict(top_bigrams['Neutral'].most_common(10)),
            'Negative': dict(top_bigrams['Negative'].most_common(10))
        }
        title = 'Top Words and Bigrams for Sentiments'
    else:  # selected_option == 'Sarcasm'
        top_words_dict = {
            'Sarcasm': dict(top_words['Sarcasm'].most_common(10))
        }
        top_bigrams_dict = {
            'Sarcasm': dict(top_bigrams['Sarcasm'].most_common(10))
        }
        title = 'Top Words and Bigrams for Sarcasm'

    # Check if the checkbox for bigrams is selected
    show_bigrams = st.checkbox("Show Bigrams")

    # Display separate charts for bigrams and words if the checkbox is selected
    if show_bigrams:
        # Create data for the bigram chart
        bigram_data = [
            go.Bar(name=key + ' - Bigrams', x=list(value.keys()), y=list(value.values()))
            for key, value in top_bigrams_dict.items()
        ]
        # Set the layout for the bigram chart
        bigram_layout = go.Layout(
            xaxis=dict(title='Bigrams'),
            yaxis=dict(title='Count'),
            width=1200,
            height=500
        )
        # Create the bigram chart
        bigram_fig = go.Figure(data=bigram_data, layout=bigram_layout)
        # Display the bigram chart using Plotly
        st.subheader(title + " - Bigrams")
        st.plotly_chart(bigram_fig)
    else:
        # Create data for the unigram chart
        unigram_data = [
            go.Bar(name=key + ' - Words', x=list(value.keys()), y=list(value.values()))
            for key, value in top_words_dict.items()
        ]
        # Set the layout for the unigram chart
        unigram_layout = go.Layout(
            xaxis=dict(title='Words'),
            yaxis=dict(title='Count'),
            width=1200,
            height=500
        )
        # Create the unigram chart
        unigram_fig = go.Figure(data=unigram_data, layout=unigram_layout)
        # Display the unigram chart using Plotly
        st.subheader(title + " - Words")
        st.plotly_chart(unigram_fig)


    
    
    
    ############ TOP 20 BIGRAM BAR CHART ############   
    


#     # Create a list to store the bigrams
#     all_bigrams = []

#     # Iterate over the preprocessed text in each row
#     for text in df['processed_text']:
#         # Tokenize the text into individual words
#         tokens = nltk.word_tokenize(text)

#         # Create the bigrams
#         bg = list(bigrams(tokens))

#         # Append the bigrams to the list
#         all_bigrams.extend(bg)

#     # Count the occurrences of each bigram
#     bigram_counts = nltk.FreqDist(all_bigrams)

#     # Convert the bigram counts to a DataFrame
#     bigram_df = pd.DataFrame(list(bigram_counts.items()), columns=['Bigram', 'Count'])

#     # Sort the bigram counts in descending order
#     sorted_bigrams = bigram_df.sort_values('Count', ascending=False)

#     # Select the top 20 bigrams
#     top_20_bigrams = sorted_bigrams.head(20)

#     # Create the chart using Altair (vertical bar chart)
#     st.subheader('Top 20 Bigram Frequency')
#     chart = alt.Chart(top_20_bigrams).mark_bar().encode(
#         x='Count',
#         y=alt.Y('Bigram', sort=alt.EncodingSortField(field='Count', order='descending')),
#         tooltip=['Bigram', 'Count']
#     ).properties(
#         width=1200,
#         height=600
#     )

#     # Display the chart
#     st.altair_chart(chart)


    
    
    
    

    ############ TOPIC MODELLING WORD CLOUD CHART ############  
    
        # Preprocess the text data
    df['processed_text'] = df['Review'].apply(preprocess_text)

    lemmatizer = WordNetLemmatizer()

    # Lemmatize the filtered_reviews column
    df['lemmatized_text'] = df['processed_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, wordnet.VERB) for word in x.split()]))

    # Preprocess the text data
    documents = df['lemmatized_text'].apply(lambda x: x.lower().split())

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(documents)

    # Convert the documents into a bag-of-words (BoW) representation
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Build the LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

      
    # Get the top words for each topic
    topic_words = {i: dict(lda_model.show_topic(i, topn=10)) for i in range(lda_model.num_topics)}

        # Define colors for wordclouds
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    # Create a wordcloud for each topic
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        topic_wordcloud = WordCloud(stopwords=stop_words,
                                    background_color='white',
                                    width=2500,
                                    height=1800,
                                    max_words=10,
                                    colormap='tab10',
                                    color_func=lambda *args, **kwargs: cols[i],
                                    prefer_horizontal=1.0)
        topic_wordcloud.generate_from_frequencies(topic_words[i], max_font_size=300)
        ax.imshow(topic_wordcloud)
        ax.set_title('Topic ' + str(i), fontdict=dict(size=16))
        ax.axis('off')

    # Adjust subplot spacing and display the wordclouds
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    st.subheader("Topic Modelling")
    st.pyplot(fig)



    
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
