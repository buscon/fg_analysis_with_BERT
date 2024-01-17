#!/usr/bin/env python3

# based on
# https://maartengr.github.io/BERTopic/getting_started/semisupervised/semisupervised.html

import configparser
import pandas as pd
import re

from classes.custom_log_bertopic import BERTopicModified, LoggerToFile
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

from hdbscan import HDBSCAN

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer

from umap import UMAP


# Create an instance of the ConfigParser class
config = configparser.ConfigParser()

# Read the contents of the `config.ini` file:
config.read('Config/config.ini')

# Access values from the configuration file:
interview_transcript = config['FILES']['interview_transcript']

# language model
language_model = config['MODELS']['embedding_model']

# UMAP parameters
n_neighbors = config['UMAP'].getint('n_neighbors')
n_components = config['UMAP'].getint('n_components')
min_dist = config['UMAP'].getfloat('min_dist')

# HDBSCAN parameters
min_cluster_size = config['HDBSCAN'].getint('min_cluster_size')

# read the transcript
# df = pd.read_csv('Data/01_transcript.txt', sep='\r')
df = pd.read_csv(interview_transcript, sep='\r')

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

for line in lines:
    match = re.match(r"(Sprecher\d+): (.+)$", line)
    if match:
        speaker = match.group(1)
        docs.append(match.group(2))
        speaker_list.add(speaker)
        sentence_to_speaker.append(int(speaker.split("Sprecher")[1]))

# Pre-calculate embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(language_model)
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# Preventing Stochastic Behavior
umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                  min_dist=min_dist, metric='cosine', random_state=42)

# Controlling Number of Topics
hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
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
#    log_to_file=True, log_file_path="my_log_file.log"
)

topics, probs = topic_model.fit_transform(docs, y=sentence_to_speaker)

new_topics = topic_model.reduce_outliers(docs, topics)

print(topic_model.get_topic_info())
print(topic_model.get_topic_info().Name)


# Run the visualization with the original embeddings
topic_model.visualize_documents(docs,
                                embeddings=embeddings).write_image("figs/01_semi_supervised.png")

# Reduce dimensionality of embeddings, this step is optional but much faster to
# perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0,
                          metric='cosine').fit_transform(embeddings)
topic_model.visualize_documents(docs,
                                reduced_embeddings=reduced_embeddings).show()
