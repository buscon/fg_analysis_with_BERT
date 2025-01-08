#!/usr/bin/env python3
import os
import pandas as pd
import re
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# Load all transcripts from the 'Data' folder
docs = []

for filename in os.listdir('Data'):
    if filename.endswith(".txt"):
        with open(os.path.join('Data', filename), 'r', encoding='utf-8') as file:
            for line in file:
                # Updated regex to handle varying spaces between "Sprecher" and the number
                match = re.match(r"^\s*(Sprecher\s*\d+):\s*(.+)$", line.strip())
                if match:
                    docs.append(match.group(2).strip())
                else:
                    print(f"Unmatched line in {filename}: {line}")

# Pre-calculate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# Define zero-shot topics
zeroshot_topic_list = ["Feeling", "Perception", "Touch", "Sound"]

# Set up models
umap_model = UMAP(n_neighbors=12, n_components=3, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, max_df=0.75, ngram_range=(1, 1))
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Fit the BERTopic model using zero-shot topics
topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model,
                       min_topic_size=10, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model,
                       zeroshot_topic_list=zeroshot_topic_list, zeroshot_min_similarity=0.25,
                       representation_model=KeyBERTInspired())

topics, probs = topic_model.fit_transform(docs, embeddings)

# Output results
print("get_topic_info:", topic_model.get_topic_info())
topic_model.visualize_documents(docs, embeddings=embeddings).write_html("figs/05_zero_shot.html")

# ** Improved Visualization: Use BERTopic's built-in visualization **
fig = topic_model.visualize_topics()
fig.write_image("figs/05_zero_shot.png")

# Save the topic info to a CSV file
topic_info = topic_model.get_topic_info()
topic_info.to_csv('output/05_zero_shot_topics_all.csv', index=False)

