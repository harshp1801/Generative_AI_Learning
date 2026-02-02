import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}
model = load_model('simple_rnn_imdb.h5')

def user_input(comment):
    txt_process = comment.lower().split()
    encoded = [word_index.get(word,2) + 3 for word in txt_process]
    padded_txt = sequence.pad_sequences([encoded],maxlen = 500)
    return padded_txt
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,"?") for i in encoded_review])


st.set_page_config(
    page_title = "Movie Review Sentiment Analyzer",
    page_icon = "💬",
    layout = "centered"
)
st.title("Movie Review Sentiment Analyzer")
st.write("Please Enter your review to analyze sentiment")
review  = st.text_area("✍️ Write your review here")
if st.button("Analyze Sentiment"):
    preprocessed_txt = user_input(review)
    predictions = model.predict(preprocessed_txt)
    sentiment = "Positive" if predictions[0][0]>0.5 else "Negative"
    if sentiment == "Positive":
        st.success(sentiment)
    elif sentiment == "Negative":
        st.error(sentiment)
    else:
        st.info(sentiment)
    st.metric("Sentiment Score (Polarity)",round(predictions[0][0],3))
else:
    st.write("Please Enter your review to analyze sentiment")    





