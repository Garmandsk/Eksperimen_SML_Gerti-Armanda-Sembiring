# Data manipulation and Numerical computing
import re  # text processing

import nltk  # Natural Language Processing NLP
import pandas as pd
from nltk.corpus import stopwords  # remove common words (like 'the', 'a', 'is')
from nltk.stem import (
    PorterStemmer,  # reduce words to their root form ("running" -> "run")
)

# Downloading NLTK resources
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    """Text preprocessing"""
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"http\S+", "URL", text)  # Replace URLs
    text = re.sub(r"\d+", "NUMBER", text)  # Replace numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters, number
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace

    # Tokenize, hapus stopwords, dan terapkan stemming secara bersamaan (List Comprehension)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]

    return " ".join(words)


def preprocess(df):
    # Remove rows with non-standard categories
    print(f"Before cleaning categories: {df.shape[0]} rows")
    df = df[df["Category"].isin(["ham", "spam"])]
    print(f"After cleaning categories: {df.shape[0]} rows")

    # Remove duplicate rows
    print(f"Before removing duplicates: {df.shape[0]} rows")
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape[0]} rows")

    # Apply cleaning
    df["cleaned_message"] = df["Message"].apply(preprocess_text)
    print("1")
    print(df.head())

    return df


file_path = "../email_raw.csv"

# Membaca dataset
df = pd.read_csv(file_path)

df = preprocess(df)
df.to_csv("email_preprocessing.csv", index=False)
