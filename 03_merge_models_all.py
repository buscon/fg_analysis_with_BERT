#!/usr/bin/env python3

import os
import re
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
import numpy as np

# Load transcripts from all three files
transcripts = []
for i in range(1, 4):
    with open(f"Data/Transcript_0{i}_Reformatted.txt", "r", encoding="utf-8") as file:
        lines = file.read().splitlines()
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
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = {speaker: embedding_model.encode(sentences, show_progress_bar=True)
              for speaker, sentences in speaker_lists.items()}

# UMAP and HDBSCAN configuration
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=42)

hdbscan_model = HDBSCAN(min_cluster_size=7, metric="euclidean",  # Increased min_cluster_size for coarser clusters
                        cluster_selection_method="eom", prediction_data=True)

# Vectorizer and representation setup
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, max_df=0.75,
                                   ngram_range=(1, 2))
representation_model = MaximalMarginalRelevance(diversity=0.5)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

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
        verbose=True
    ).fit(sentences)
    print(f"Number of topics for {speaker}: {len(topic_models[speaker].get_topic_info())}")

# Merge topic models with stricter merging criteria
merged_model = BERTopic.merge_models(list(topic_models.values()), min_similarity=0.92)

print(f"Number of merged topics before limiting: {len(merged_model.get_topic_info())}")

# Function to limit topics to a maximum number
def limit_topics(topic_model, max_topics):
    topic_info = topic_model.get_topic_info()
    if len(topic_info) > max_topics:
        topics_to_keep = topic_info.sort_values(by='Count', ascending=False).head(max_topics)
        filtered_topic_ids = topics_to_keep['Topic'].tolist()

        # Keep only the top topics
        topic_model.topic_info = topic_info[topic_info['Topic'].isin(filtered_topic_ids)]
    return topic_model

# Limit the number of topics to a range of 20–25
merged_model = limit_topics(merged_model, max_topics=25)
print(f"Number of merged topics after limiting: {len(merged_model.get_topic_info())}")

# Save final refined topics
topic_info = merged_model.get_topic_info()
filtered_topics = topic_info[topic_info['Topic'] > 0]
filtered_topics.to_csv("output/03_refined_topics.csv", index=False)

# Flatten embeddings to match sentences for visualization
flattened_embeddings = np.vstack(list(embeddings.values()))

# Visualize the merged topics
try:
    merged_model.visualize_documents(transcripts, embeddings=flattened_embeddings).write_html("figs/03_topics.html")
except ValueError:
    print("Visualization skipped due to missing embeddings.")

