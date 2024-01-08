#!/usr/bin/env python3

import pandas as pd
from umap import UMAP
from bertopic import BERTopic

# list to store the speakers data, one list of strings per speaker
speakers = []

# loop over the dataset of each speaker and make a list of sentences
for speaker_n in range(1, 6):
    # read the dataset
    df = pd.read_csv(
        '/home/marcello/MEGAsync/Promotion/FG/01/01_transcript_stra03_sp0'
        + str(speaker_n) +
        '.txt',
        sep='\r')

    # transform the df to a list
    list_of_sentences = df.values.tolist()

    # flatten the list of list into a list of strings and remove \xa0
    speaker = []
    # convert list_of_sentences (a list of lists) into a list of strings
    for sentence in list_of_sentences:
        speaker.append(sentence[0].replace('\xa0', ''))
    speakers.append(speaker)

# Create topic models
umap_model = UMAP(n_neighbors=30, n_components=10, min_dist=0.0,
                  metric='cosine', random_state=42)

# list to store the speakers data, one list of strings per speaker
topic_models = []
for speaker_n in range(0, 5):
#    topic_model = BERTopic().fit(speakers[speaker_n])
    topic_model = BERTopic(umap_model=umap_model).fit(speakers[speaker_n])
    topic_models.append(topic_model)

# merge all the models
merged_model = BERTopic.merge_models([topic_models[0], topic_models[1],
                                      topic_models[2], topic_models[3],
                                      topic_models[4]])


print("Numbers of topics in topic_models[0]: ",
      len(topic_models[0].get_topic_info()))
print("get_topic_info topic_models[0]: ", topic_models[0].get_topic_info())
print("Numbers of topics in merged_model: ",
      len(merged_model.get_topic_info()))
print("get_topic_info merged model: ", merged_model.get_topic_info())

# print("get_topic: ", model.get_topic(0))

# print(model.get_document_info(docs))

# use a different model
# model = SentenceTransformer('all-mpnet-base-v2')

# Visualize topics
# model.visualize_topics().show()

# Visualize probabilities
# model.visualize_distribution(probs[0]).show()

# Visualize hierarchy
# model.visualize_hierarchy().show()


# Visualize Topic Terms
# model.visualize_barchart().show()


# Visualize Topic Similarity
# model.visualize_heatmap().show()


# Visualize Term Score Decline
# model.visualize_term_rank().show()
