#!/usr/bin/env python3

import configparser
import os
import re
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np

# Load transcripts from all three files
transcripts = []
for i in range(1, 4):
    df = pd.read_csv(f'Data/Transcript_0{i}_Reformatted.txt', sep='\r')
    list_of_sentences = df.values.tolist()
    lines = [sentence[0].replace('\xa0', '') for sentence in list_of_sentences]
    transcripts.extend(lines)

# Process the data to extract speakers and their sentences
speaker_lists = {}
for line in transcripts:
    match = re.match(r"(Sprecher\d+): (.+)$", line)
    if match:
        speaker = match.group(1)
        sentence = match.group(2).lstrip()
        if speaker not in speaker_lists:
            speaker_lists[speaker] = []
        speaker_lists[speaker].append(sentence)

# Pre-calculate embeddings
embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
embeddings = {speaker: embedding_model.encode(sentences, show_progress_bar=True)
              for speaker, sentences in speaker_lists.items()}

# Prevent stochastic behavior
umap_model = UMAP(n_neighbors=20, n_components=3, min_dist=0.0,
                  metric='cosine', random_state=42)

# Control number of topics
hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean',
                        cluster_selection_method='eom', prediction_data=True)

# Vectorizer setup
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, max_df=0.75,
                                   ngram_range=(1, 2))

# Reduce the impact of frequent words
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Diversify topic representation
representation_model = MaximalMarginalRelevance(diversity=0.5)

# Fit BERTopic models for each speaker
topic_models = {}
for speaker, sentences in speaker_lists.items():
    topic_models[speaker] = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        ctfidf_model=ctfidf_model,
        top_n_words=10, verbose=True, calculate_probabilities=True
    ).fit(sentences)
    print(f"Number of topics for {speaker}: {len(topic_models[speaker].get_topic_info())}")

# Merge topic models
merged_model = BERTopic.merge_models(list(topic_models.values()),
                                     min_similarity=0.9)

print(f"Number of merged topics: {len(merged_model.get_topic_info())}")

# Post-process to filter trivial topics
def filter_trivial_topics(topic_model, keywords_to_remove):
    topic_info = topic_model.get_topic_info()
    filtered_topics = topic_info[
        (topic_info['Name'].str.len() > 10) &  # Remove very short topics
        (~topic_info['Name'].str.contains('|'.join(keywords_to_remove), case=False))  # Filter noisy topics
    ]
    topic_model.topic_info = filtered_topics
    return topic_model

# Apply filtering
keywords_to_remove = ['yep', 'like', 'say', 'think', 'yeah', 'ok', 'use', 'thank', 'good', 'fine']
merged_model = filter_trivial_topics(merged_model, keywords_to_remove)

# Run visualization with original embeddings
new_docs = [sentence for sublist in speaker_lists.values() for sentence in sublist]
merged_model.visualize_documents(new_docs).write_image("figs/03_merged_model.png")

# Save filtered topic info
topic_info = merged_model.get_topic_info()
filtered_topics = topic_info[topic_info['Topic'] > 0]
filtered_topics.to_csv('output/03_refined_topics.csv', index=False)

# Flatten embeddings to match sentences for visualization
flattened_embeddings = np.vstack(list(embeddings.values()))

# Run visualization with flattened embeddings
try:
    merged_model.visualize_documents(new_docs, embeddings=flattened_embeddings).write_html("figs/03_topics.html")
except ValueError:
    print("Warning: Document visualization skipped due to empty embeddings.")

