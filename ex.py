import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = [
    ("I love this sandwich", "pos"),
    ("This is an amazing place", "pos"),
    ("I feel very good about these beers", "pos"),
    ("This is my best work", "pos"),
    ("What an awesome view", "pos"),
    ("I do not like this restaurant", "neg"),
    ("I am tired of this stuff", "neg"),
    ("I can't deal with this", "neg"),
    ("He is my sworn enemy", "neg"),
    ("My boss is horrible", "neg"),
    ("This is an awesome place", "pos"),
    ("I do not like the taste of this juice", "neg"),
    ("I love to dance", "pos"),
    ("I am sick and tired of this place", "neg"),
    ("What a great holiday", "pos"),
    ("That is a bad locality to stay", "neg"),
    ("We will have good fun tomorrow", "pos"),
    ("I went to my enemy's house today", "neg")
]

# Preprocess the data
def preprocess_data(data):
    df = pd.DataFrame(data, columns=['text', 'label'])
    df['text'] = df['text'].str.lower()
    return df

# Streamlit app
def main():
    st.title("Naive Bayes Document Classification")

    # Display dataset
    df = preprocess_data(data)
    st.write("Dataset:")
    st.write(df)

    # Train the model
    if st.button("Train Model"):
        st.write("Training model...")
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(df['text'])
        clf = MultinomialNB()
        clf.fit(X_train_counts, df['label'])
        st.success("Model trained successfully!")

        # New document input
        st.subheader("Classify New Document")
        doc = st.text_input("Enter new document:")
        if st.button("Classify"):
            X_new_counts = count_vect.transform([doc])
            predicted_class = clf.predict(X_new_counts)
            st.write(f"Predicted class: {predicted_class[0]}")

if __name__ == "__main__":
    main()
