#!/usr/bin/env python3

import configparser
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = {'yeah', 'yep', 'think', 'say', 'like', 'really', 'ok', 'use', 'im'}  # Custom list of overused words to remove
stop_words.update(custom_stopwords)
stop_words.discard('experience')  # Keep 'experience' in the analysis

# Load the configuration file from the specified location
config = configparser.ConfigParser()
config.read('Config/config.ini')

# Function to preprocess text (lemmatization, stopword removal)
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase the text
    tokens = text.split()  # Tokenize by whitespace
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Function to load transcripts and retain speaker prefixes
def load_transcripts_with_prefix(directory):
    transcripts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                speaker_blocks = re.findall(r'(Sprecher\d+:\s*[^\n]+)', content)
                for block in speaker_blocks:
                    speaker, text = block.split(':', 1)
                    processed_text = preprocess_text(text.strip())
                    transcripts.append(f"{processed_text}")
    return transcripts

# Load transcripts
transcripts = load_transcripts_with_prefix('./Data')

# Prepare data by splitting into sentences
sentences = [sentence.strip() for transcript in transcripts for sentence in transcript.splitlines() if sentence.strip()]

# Load the embedding model
embedding_model = SentenceTransformer(config['MODELS']['embedding_model'])
embeddings = embedding_model.encode(sentences, show_progress_bar=True)

# Handle zero-size embedding case
if embeddings.shape[0] == 0:
    embeddings = np.zeros((1, embeddings.shape[1]))  # Avoid zero-size array error

# Improved vectorizer for better context capture
vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=1, max_df=0.75)

# Define BERTopic model
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.2, metric='cosine')
topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    embedding_model=embedding_model,
    umap_model=umap_model,
    calculate_probabilities=True,
    verbose=True,
    min_topic_size=10
)

# Fit the model
topics, probs = topic_model.fit_transform(sentences, embeddings)

# Post-process to remove trivial topics
def filter_trivial_topics(topic_model):
    topic_info = topic_model.get_topic_info()
    filtered_topics = topic_info[topic_info['Name'].str.len() > 10]
    topic_model.topic_info = filtered_topics
    return topic_model

# Apply filtering
topic_model = filter_trivial_topics(topic_model)

# Visualization and outputs
fig = topic_model.visualize_topics()
fig.write_image("figs/01_prefix_topics.png")

try:
    if embeddings.shape[0] > 1:
        umap_embeddings = umap_model.fit_transform(embeddings)
        plt.figure(figsize=(10, 7))
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=topics, cmap='Spectral', s=10)
        plt.title("UMAP projection of embeddings and BERTopic clusters")
        plt.savefig("figs/01_prefix_umap.png")
    else:
        print("Warning: UMAP visualization skipped due to insufficient embeddings.")
except ValueError:
    print("Warning: UMAP failed due to empty or invalid embeddings.")

# Save topic info
topic_info = topic_model.get_topic_info()
filtered_topics = topic_info[topic_info['Topic'] > 0]
filtered_topics.to_csv('output/01_refined_topics.csv', index=False)

# Print refined topics
for topic_id, topic in topic_model.get_topics().items():
    print(f"Topic {topic_id}: {topic}")

# Document visualization
try:
    topic_model.visualize_documents(sentences, embeddings=embeddings).write_html("figs/01_refined_documents.html")
except ValueError:
    print("Warning: Document visualization skipped due to empty embeddings.")

