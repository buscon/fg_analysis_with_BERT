import pandas as pd

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# Change the default renderer to svg import plotly.io as pio
# pio.renderers.default = "svg"

df = pd.read_csv('/home/marcello/MEGAsync/Promotion/FG/01/01_transcript.txt',
                 sep='\r')

list_of_sentences = df.values.tolist()

docs = []

for sentence in list_of_sentences:
    docs.append(sentence[0].replace('\xa0', ''))

# Pre-calculate embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("thenlper/gte-small")
# embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# We define a number of topics that we know are in the documents
zeroshot_topic_list = ["Feeling", "Perception", "Touch", "Sound"]


# We fit our model using the zero-shot topics
# and we define a minimum similarity. For each document,
# if the similarity does not exceed that value, it will be used
# for clustering instead.
topic_model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=15,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=.85,
    representation_model=KeyBERTInspired()
)

topics, probs = topic_model.fit_transform(docs, embeddings)

print("get_topic_info: ", topic_model.get_topic_info())

# Run the visualization with the original embeddings
topic_model.visualize_documents(docs, embeddings=embeddings).show()

# Reduce dimensionality of embeddings, this step is optional but much faster to
# perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0,
                          metric='cosine').fit_transform(embeddings)
topic_model.visualize_documents(docs,
                                reduced_embeddings=reduced_embeddings).show()
