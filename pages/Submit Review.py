import streamlit as st
import nltk
import string
import re
import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
from datetime import datetime

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

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


text_cleaning = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stemmer = PorterStemmer()

model_sentiment = joblib.load('model_sentiment.pkl')
vectorizer_sentiment = joblib.load('vectorizer_sentiment.pkl')

model_sarcasm = joblib.load('model_sarcasm.pkl')
vectorizer_sarcasm = joblib.load('vectorizer_sarcasm.pkl')

#define a function to remove the stopwords and combine the remaining words as the sentence
def remove_stopwords(review):
    tokens = nltk.tokenize.word_tokenize(review)
    filtered_review = [token for token in tokens if not token.lower() in filtered_stop_words]
    return ' '.join(filtered_review)

def preprocess_text(text):
    text = re.sub(text_cleaning, ' ', str(text).lower()).strip()
    text = stemmer.stem(str(text))
    # Tokenization and removing stopwords
    text = remove_stopwords(text)
    return text

def main():
    st.title("Review Submission")
    
    # Input fields
    name = st.text_input("Name")
    review = st.text_area("Review")
    rating = st.slider("Rating", min_value=1, max_value=5)
    
    # Submit button
    if st.button("Submit"):
        # Process the submitted data
        process_submission(name, review, rating)
        st.success("Review submitted successfully!")

def process_submission(name, review, rating):

    preprocessed_sentence = preprocess_text(review)

    # Wrap the string in a list
    documents = [preprocessed_sentence]

    # Predict sarcasm prediction
    vectorized_input = vectorizer_sarcasm.transform(documents)  
    predicted_label = model_sarcasm.predict(vectorized_input)[0]

    # Perform sentiment prediction
    sentence_vectorized = vectorizer_sentiment.transform(documents)
    prediction = model_sentiment.predict(sentence_vectorized)[0]

    # Append the input, rating, and current date to the CSV file
    with open('user_reviews.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(["Name", "Rating", "Review", "Sarcasm", "Sentiment", "Date"])  # Write column headers if file is empty
        writer.writerow([name, rating, review, predicted_label, prediction, datetime.now()])

    st.success("Review submitted successfully!")

if __name__ == "__main__":
    main()
