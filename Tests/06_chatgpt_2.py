#!/usr/bin/env python3

import configparser
import pandas as pd
import re
import os
from datetime import datetime
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# Load configurations from config.ini
config = configparser.ConfigParser()
config.read('Config/config.ini')

# Directory containing transcript files
data_folder = config['FILES']['data_folder']

# Language model to be used
language_model = config['MODELS']['embedding_model']

# UMAP parameters
n_neighbors = config['UMAP'].getint('n_neighbors')
n_components = config['UMAP'].getint('n_components')
min_dist = config['UMAP'].getfloat('min_dist')

# HDBSCAN parameters
min_cluster_size = config['HDBSCAN'].getint('min_cluster_size')

docs = []
sentence_to_speaker = []
speaker_list = set()

# Read and process each transcript file in the Data folder
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath, sep='\r', engine='python')
        list_of_sentences = df.values.tolist()

        # Process each line to extract speaker and text
        for sentence in list_of_sentences:
            line = sentence[0].replace('\xa0', '')
            match = re.match(r"(Sprecher\d+): (.+)$", line)
            if match:
                speaker = match.group(1)
                text = match.group(2)
                # Simple text preprocessing: lowercase, remove non-alphabetic chars, etc.
                text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                docs.append(text)
                speaker_list.add(speaker)
                sentence_to_speaker.append(int(speaker.split("Sprecher")[1]))

# Load the embedding model and compute embeddings
embedding_model = SentenceTransformer(language_model)
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# Configure UMAP for dimensionality reduction
umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                  min_dist=min_dist, metric='cosine', random_state=42)

# Configure HDBSCAN for clustering
hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                        cluster_selection_method='eom', prediction_data=True)

# Customize stopwords list by adding conversational fillers and irrelevant terms
custom_stopwords = [
    "yeah", "don", "ve", "know", "think", "like", "get",
    "really", "maybe", "respond", "well", "uh", "um", "okay",
    "actually", "just", "right", "im", "thats"
    "it", "this", "some", "to", "thats"
]

# Configure the vectorizer model and Class-TfIdf
vectorizer_model = CountVectorizer(stop_words=custom_stopwords, min_df=2, max_df=0.7,
                                   ngram_range=(1, 2))  # Focus on bigrams and trigrams
# reduce the impact of frequent words
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Use KeyBERTInspired for topic representation with higher diversity
representation_model = MaximalMarginalRelevance(diversity=0.9)


# Create a unique log file name using the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Initialize the BERTopic model with custom configurations
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    ctfidf_model=ctfidf_model,
    top_n_words=5,  # Reduced top_n_words to refine topic labeling
    verbose=True, calculate_probabilities=True
)

# Fit the model and transform the documents to get topics
topics, probs = topic_model.fit_transform(docs, y=sentence_to_speaker)

# Skip topic reduction and use the initially extracted topics

# Display the topic information
print(topic_model.get_topic_info())
print(topic_model.get_topic_info().Name)

# Visualize the documents in 2D space with original embeddings
topic_model.visualize_documents(docs, embeddings=embeddings).write_image("figs/01_semi_supervised.png")

# Optionally, reduce dimensionality of embeddings for faster visualization
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0,
                          metric='cosine').fit_transform(embeddings)
topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings).show()

