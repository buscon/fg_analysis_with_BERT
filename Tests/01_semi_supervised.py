#!/usr/bin/env python3

# based on
# https://maartengr.github.io/BERTopic/getting_started/semisupervised/semisupervised.html

import configparser
import pandas as pd
import re

from datetime import datetime
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

# Updated regex to handle varying spaces between "Sprecher" and the number
for line in lines:
    match = re.match(r"^\s*(Sprecher\s*\d+):\s*(.+)$", line.strip())
    if match:
        speaker = match.group(1).strip()
        docs.append(match.group(2).strip())
        speaker_list.add(speaker)
        sentence_to_speaker.append(int(speaker.split("Sprecher")[1]))

# Pre-calculate embeddings
embedding_model = SentenceTransformer(language_model)
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# Preventing Stochastic Behavior
umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                  min_dist=min_dist, metric='cosine', random_state=42)

# Controlling Number of Topics
hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                        cluster_selection_method='eom', prediction_data=True)

vectorizer_model = CountVectorizer(stop_words="english", min_df=2, max_df=0.75,
                                   ngram_range=(1, 2))

# reduce the impact of frequent words
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Diversify topic representation
representation_model = MaximalMarginalRelevance(diversity=0.7)

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

topic_model = BERTopicModified(
    embedding_model=embedding_model, umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    ctfidf_model=ctfidf_model,
    top_n_words=10, verbose=True, calculate_probabilities=True,
    log_to_file=True, log_file_path="Logs/" + current_datetime + "_my_log_file.log"
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

# Save the topic info to a CSV file
topic_info = topic_model.get_topic_info()
topic_info.to_csv('output/01_semi_supervised_topic.csv', index=False)

