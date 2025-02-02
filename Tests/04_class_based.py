#!/usr/bin/env python3

# based on
# https://maartengr.github.io/BERTopic/getting_started/topicsperclass/topicsperclass.html

import pandas as pd
import re

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


# read the transcript
df = pd.read_csv('Data/Transcript_01_Reformatted.txt', sep='\r')

# make the dataframe into a list of sentences
list_of_sentences = df.values.tolist()
lines = []

for sentence in list_of_sentences:
    lines.append(sentence[0].replace('\xa0', ''))

# process the data, create two lists:
# list of speakers
# list of which speaker of each sentence

speaker_list = set()  # Using a set to store unique speakers
sentence_to_speaker = []
docs = []

# process the data, create three lists:
#   1. list of sentences
#   2. list of speakers

speaker_list = []

for line in lines:
    match = re.match(r"(Sprecher\d+): (.+)$", line)
    if match:
        speaker = match.group(1)
        docs.append(match.group(2))
        speaker_list.append(speaker)

# Pre-calculate embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# Preventing Stochastic Behavior
umap_model = UMAP(n_neighbors=15, n_components=3, min_dist=0.0,
                  metric='cosine', random_state=42)

# Controlling Number of Topics
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean',
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

topic_model = BERTopic(
    # Pipeline models
    embedding_model=embedding_model, umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    ctfidf_model=ctfidf_model,
    # Hyperparameters
    top_n_words=10, verbose=True, calculate_probabilities=True
)

topics, probs = topic_model.fit_transform(docs, embeddings)

new_topics = topic_model.reduce_outliers(docs, topics)

print(topic_model.get_topic_info())
print(topic_model.get_topic_info().Name)

topics_per_class = topic_model.topics_per_class(docs, classes=speaker_list)

# topic_model.visualize_topics_per_class(topics_per_class,
#                                        top_n_topics=10).show()
topic_model.visualize_documents(docs,
                                embeddings=embeddings).write_image("figs/04_class_based.png")
# .write_html("figs/04_class_based.html")


# Run the visualization with the original embeddings
# topic_model.visualize_documents(docs, embeddings=embeddings).show()

# Reduce dimensionality of embeddings, this step is optional but much faster to
# perform iteratively:
# reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0,
#                           metric='cosine').fit_transform(embeddings)
# topic_model.visualize_documents(docs,
#                                 reduced_embeddings=reduced_embeddings).show()
