import streamlit as st
import pandas as pd

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
    df['text'] = df['text'].str.lower().str.split()
    return df

# Train Naive Bayes model
def train_naive_bayes(df):
    num_pos = (df['label'] == 'pos').sum()
    num_neg = len(df) - num_pos
    total_docs = len(df)
    
    p_pos = num_pos / total_docs
    p_neg = num_neg / total_docs
    
    word_counts_pos = {}
    word_counts_neg = {}
    
    for index, row in df.iterrows():
        for word in row['text']:
            if row['label'] == 'pos':
                word_counts_pos[word] = word_counts_pos.get(word, 0) + 1
            else:
                word_counts_neg[word] = word_counts_neg.get(word, 0) + 1
    
    vocab_size = len(set(df['text'].sum()))
    p_word_given_pos_smooth = {word: (count + 1) / (num_pos + vocab_size) for word, count in word_counts_pos.items()}
    p_word_given_neg_smooth = {word: (count + 1) / (num_neg + vocab_size) for word, count in word_counts_neg.items()}
    
    return p_pos, p_neg, p_word_given_pos_smooth, p_word_given_neg_smooth

# Classify a document
def classify_document(document, p_pos, p_neg, p_word_given_pos, p_word_given_neg):
    p_pos_given_doc = p_pos
    p_neg_given_doc = p_neg
    
    for word in document:
        p_pos_given_doc *= p_word_given_pos.get(word, 1 / (len(p_word_given_pos) + 1))
        p_neg_given_doc *= p_word_given_neg.get(word, 1 / (len(p_word_given_neg) + 1))
    
    return 'pos' if p_pos_given_doc > p_neg_given_doc else 'neg'

# Streamlit app
def main():
    st.title("Naive Bayes Document Classification")

    # Display dataset
    df = preprocess_data(data)
    st.write("Dataset:")
    st.write(df)

    # Train the model
    if st.button("Train Model"):
        p_pos, p_neg, p_word_given_pos, p_word_given_neg = train_naive_bayes(df)
        st.success("Model trained successfully!")

        # New document input
        st.subheader("Classify New Document")
        doc = st.text_input("Enter new document:")
        if st.button("Classify"):
            doc = doc.lower().split()
            predicted_class = classify_document(doc, p_pos, p_neg, p_word_given_pos, p_word_given_neg)
            st.write(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
