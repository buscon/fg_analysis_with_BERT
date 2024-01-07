#!/usr/bin/env python3


import pandas as pd

from bertopic import BERTopic

# Change the default renderer to svg
# import plotly.io as pio
# pio.renderers.default = "svg"

df = pd.read_csv(
    '/home/marcello/MEGAsync/Promotion/FG/01/transcript_text_only.txt',
    sep='\r')

list_of_sentences = df.values.tolist()
docs = []

for sentence in list_of_sentences:
    docs.append(sentence[0].replace('\xa0', ''))

model = BERTopic(calculate_probabilities=True)
topics, probs = model.fit_transform(docs)

print("get_topic_info: ", model.get_topic_info())

print("get_topic: ", model.get_topic(0))

# print(model.get_document_info(docs))

# use a different model
# model = SentenceTransformer('all-mpnet-base-v2')

# Visualize topics
model.visualize_topics().show()

# Visualize probabilities
model.visualize_distribution(probs[0]).show()

# Visualize hierarchy
model.visualize_hierarchy().show()


# Visualize Topic Terms
# model.visualize_barchart().show()


# Visualize Topic Similarity
# model.visualize_heatmap().show()


# Visualize Term Score Decline
# model.visualize_term_rank().show()
