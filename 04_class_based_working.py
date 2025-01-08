#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import re
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import matplotlib.pyplot as plt

# Load all transcripts from the 'Data' folder
docs = []
speaker_list = []

for filename in os.listdir('Data'):
    if filename.endswith(".txt"):
        with open(os.path.join('Data', filename), 'r', encoding='utf-8') as file:
            for line in file:
                match = re.match(r"^\s*(Sprecher\s*\d+):\s*(.+)$", line.strip())
                if match:
                    speaker_list.append(match.group(1).strip())
                    docs.append(match.group(2).strip())
                else:
                    print(f"Unmatched line in {filename}: {line}")

# Pre-calculate embeddings
embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
embeddings = embedding_model.encode(docs, show_progress_bar=True)
embeddings = np.array(embeddings)  # Convert embeddings to a NumPy array

# Set up models
umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, max_df=0.75, ngram_range=(1, 2))
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
representation_model = MaximalMarginalRelevance(diversity=0.5)

# Fit the BERTopic model
topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model,
                       vectorizer_model=vectorizer_model, representation_model=representation_model, ctfidf_model=ctfidf_model,
                       top_n_words=10, verbose=True, calculate_probabilities=True)

topics, probs = topic_model.fit_transform(docs, embeddings)

# Check and print lengths of arrays for debugging
print(f"Length of docs: {len(docs)}")
print(f"Length of speaker_list: {len(speaker_list)}")
print(f"Length of topics: {len(topics)}")

# Adjust the lists to ensure they have matching lengths (filter out -1 noise topics)
valid_indices = [i for i, topic in enumerate(topics) if topic != -1]

docs = [docs[i] for i in valid_indices]
speaker_list = [speaker_list[i] for i in valid_indices]
topics = [topics[i] for i in valid_indices]
embeddings = embeddings[valid_indices]  # Ensure embeddings match valid indices

# Update the model's topics_ with the filtered topics
topic_model.topics_ = topics

# Check lengths again after filtering
print(f"After filtering, length of docs: {len(docs)}")
print(f"After filtering, length of speaker_list: {len(speaker_list)}")
print(f"After filtering, length of topics: {len(topics)}")

# Class-based topic analysis
topics_per_class = topic_model.topics_per_class(docs, classes=speaker_list)

# Output results
print(topic_model.get_topic_info())
topic_model.visualize_documents(docs, embeddings=embeddings).write_html("figs/04_class_based.html")

# ** Custom UMAP dimensionality reduction for visualization **
reduced_embeddings = umap_model.fit_transform(embeddings)

# Plot the clusters using the custom UMAP-reduced embeddings
plt.figure(figsize=(10, 7))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=topics, cmap='Spectral', s=10)
plt.title("UMAP projection of sentence embeddings and BERTopic clusters")
plt.savefig("figs/04_class_based_umap.png")
plt.show()

# Save the topic info to a CSV file
topic_info = topic_model.get_topic_info()
topic_info.to_csv('output/04_class_based_all.csv', index=False)

