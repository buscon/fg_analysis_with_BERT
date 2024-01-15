import pandas as pd

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN


# based on
# https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html

df = pd.read_csv('/home/marcello/MEGAsync/Promotion/FG/01/01_transcript.txt',
                 sep='\r')

list_of_sentences = df.values.tolist()

docs = []

for sentence in list_of_sentences:
    docs.append(sentence[0].replace('\xa0', ''))

# Pre-calculate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# embedding_model = SentenceTransformer("thenlper/gte-small")
# embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# We define a number of topics that we know are in the documents
zeroshot_topic_list = ["Feeling", "Perception", "Touch", "Sound"]

# Preventing Stochastic Behavior
umap_model = UMAP(n_neighbors=12, n_components=3, min_dist=0.0,
                  metric='cosine', random_state=42)

# Controlling Number of Topics
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean',
                        cluster_selection_method='eom', prediction_data=True)


# Here, we will ignore English stopwords and infrequent words. Moreover, by
# increasing the n-gram range we will consider topic representations that are
# made up of one or two words.
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, max_df=0.75,
                                   ngram_range=(1, 1))

# reduce the impact of frequent words
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# We fit our model using the zero-shot topics
# and we define a minimum similarity. For each document,
# if the similarity does not exceed that value, it will be used
# for clustering instead.
#    representation_model=representation_model,
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    min_topic_size=10,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=.25,
    representation_model=KeyBERTInspired()
)

topics, probs = topic_model.fit_transform(docs, embeddings)

print("get_topic_info: \n", topic_model.get_topic_info())
print("get_topic_info Names: \n", topic_model.get_topic_info().Name)

# Run the visualization with the original embeddings
topic_model.visualize_documents(docs,
                                embeddings=embeddings).write_image("figs/05_zero_shot.png")
# .write_html("figs/05_zero_shot.html")

# Reduce dimensionality of embeddings, this step is optional but much faster to
# perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0,
                          metric='cosine').fit_transform(embeddings)
topic_model.visualize_documents(docs,
                                reduced_embeddings=reduced_embeddings).show()
