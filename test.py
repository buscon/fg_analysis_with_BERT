import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

class TextAnalysis:
    def __init__(self, text):
        self.text = text
        self.docs = []

    def preprocess(self):
        # Change the default renderer to svg import plotly.io as pio
        # pio.renderers.default = "svg"
        df = pd.read_csv(self.text, sep='\r')
        list_of_sentences = df.values.tolist()
        for sentence in list_of_sentences:
            self.docs.append(sentence[0].replace('\xa0', ''))
        return self.docs

    def embeddings(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # or "WhereIsAI/UAE-Large-V1"
        self.embeddings = self.embedding_model.encode(self.docs, show_progress_bar=True)
        return self.embeddings

    def visualize(self):
        # Preventing Stochastic Behavior
        umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
        # Controlling Number of Topics
        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        # Here, we will ignore English stopwords and infrequent words. Moreover, by increasing the n-gram range we will consider topic representations that are made up of one or two words.
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 1))
        # reduce the impact of frequent words
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        # KeyBERT-Inspired model to reduce the appearance of stop words.
        representation_model = KeyBERTInspired()
        # Diversify topic representation
        representation_model = MaximalMarginalRelevance(diversity=0.5)
        topic_model = BERTopic(
            # Pipeline models
            embedding_model=self.embedding_model, umap_model=umap_model,
            hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            ctfidf_model=ctfidf_model,
            # Hyperparameters
            top_n_words=10, verbose=True, calculate_probabilities=True)

        topics, probs = topic_model.fit_transform(self.docs, self.embeddings)

        # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:

        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(self.embeddings)

        topic_model.visualize_documents(self.docs, reduced_embeddings=reduced_embeddings).show()



# create a new instance of the TextAnalysis class and pass in your text data as an argument
analyzer = TextAnalysis("/home/marcello/MEGAsync/Promotion/FG/01/01_transcript.txt")

# call the preprocess method to clean and tokenize the text data
cleaned_data = analyzer.preprocess()

# call the embeddings method to generate embeddings for the cleaned data
embeddings = analyzer.embeddings()

# call the visualize method to create a topic model and visualize the results
analyzer.visualize()
