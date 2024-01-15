#!/usr/bin/env python3

# based on
# https://maartengr.github.io/BERTopic/getting_started/merge/merge.html

import pandas as pd
import re

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


df = pd.read_csv('Data/01_transcript.txt', sep='\r')

# make the dataframe into a list of sentences
list_of_sentences = df.values.tolist()

lines = []

for sentence in list_of_sentences:
    lines.append(sentence[0].replace('\xa0', ''))

# process the data, create three lists:
#   1. list of sentences
#   2. list of speakers
#   3. dict of which speaker spoke which sentence

# dict of Speakers. each value will be a list of sentences of each speaker.
speaker_lists = {f"Sprecher{i}": [] for i in range(1, 6)}

for line in lines:
    match = re.match(r"(Sprecher\d+): (.+)$", line)
    if match:
        speaker = match.group(1)
        sentence = match.group(2).lstrip()  # Remove leading spaces
        speaker_lists[speaker].append(sentence)

# Pre-calculate embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
embeddings = dict.fromkeys(speaker_lists.keys())
for speaker in speaker_lists:
    embeddings[speaker] = embedding_model.encode(speaker_lists[speaker],
                                                 show_progress_bar=True)

# Preventing Stochastic Behavior
umap_model = UMAP(n_neighbors=20, n_components=3, min_dist=0.0,
                  metric='cosine', random_state=42)

# Controlling Number of Topics
hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean',
                        cluster_selection_method='eom', prediction_data=True)

# Here, we will ignore English stopwords and infrequent words. Moreover, by
# increasing the n-gram range we will consider topic representations that are
# made up of one or two words.
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, max_df=0.75,
                                   ngram_range=(1, 2))

# reduce the impact of frequent words
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# KeyBERT-Inspired model to reduce the appearance of stop words.
# representation_model = KeyBERTInspired()

# Diversify topic representation
representation_model = MaximalMarginalRelevance(diversity=0.5)

topic_models = dict.fromkeys(speaker_lists.keys())

for speaker in topic_models:
    topic_models[speaker] = BERTopic(
        # Pipeline models
        embedding_model=embedding_model, umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        ctfidf_model=ctfidf_model,
        # Hyperparameters
        top_n_words=10, verbose=True, calculate_probabilities=True
    ).fit(speaker_lists[speaker])
    print("Number of topics for {}: {}".format(speaker,
                                               len(topic_models[speaker].get_topic_info())))
    print("Topics for {}: {}".format(speaker,
                                               topic_models[speaker].get_topic_info()))

# (dis)similarity can be tweaked using the min_similarity parameter.
# Increasing this value will increase the change of adding new topics. In
# contrast, decreasing this value will make it more strict and threfore
# decrease the change of adding new topics.
merged_model = BERTopic.merge_models(list(topic_models.values()),
                                     min_similarity=0.9)

print("Number of merged topics: {}".format(len(merged_model.get_topic_info())))
print("merged topics: {}".format(merged_model.get_topic_info()))
print("merged topics names: {}".format(merged_model.get_topic_info().Name))

# Run the visualization with the original embeddings
new_docs = [item for sublist in list(speaker_lists.values()) for item in sublist]

merged_model.visualize_documents(new_docs).write_image("figs/03_merged_model.png")
# merged_model.visualize_documents(new_docs).write_html("figs/03_merged_model.html")


