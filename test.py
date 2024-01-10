import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import matplotlib.pyplot as plt

class TopicModeling:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, sep='\r')
        self.list_of_sentences = self.df.values.tolist()
        self.docs = [sentence[0].replace('\xa0', '') for sentence in self.list_of_sentences]
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embedding_model.encode(self.docs, show_progress_bar=True)
        self.umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
        self.hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean',
                                     cluster_selection_method='eom', prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 1))
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.representation_model = MaximalMarginalRelevance(diversity=0.5)
        self.topic_model = None

    def _initialize_topic_model(self):
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model, umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model, vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model, ctfidf_model=self.ctfidf_model,
            top_n_words=10, verbose=True, calculate_probabilities=True
        )

    def fit_transform(self):
        if self.topic_model is None:
            self._initialize_topic_model()
        self.topics, self.probs = self.topic_model.fit_transform(self.docs, self.embeddings)

    def visualize_documents(self, use_reduced_embeddings=False):
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        if use_reduced_embeddings:
            reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine')\
                .fit_transform(self.embeddings)
            self.topic_model.visualize_documents(self.docs, reduced_embeddings=reduced_embeddings).show()
        else:
            self.topic_model.visualize_documents(self.docs, embeddings=self.embeddings).show()

    def get_topic_info(self):
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        return self.topic_model.get_topic_info()

    def get_topic(self, topic_id):
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        return self.topic_model.get_topic(topic_id)

    def get_document_info(self):
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        return self.topic_model.get_document_info(self.docs)

    def _visualize(self, plot_object, show=True, save_path=None):
        if show:
            plot_object.show()
        if save_path:
            plot_object.write_image(save_path)

    def visualize_topics(self, show=True, save_path=None):
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        plot_object = self.topic_model.visualize_topics()
        self._visualize(plot_object, show, save_path)

    def visualize_distribution(self, prob, show=True, save_path=None):
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        plot_object = self.topic_model.visualize_distribution(prob)
        self._visualize(plot_object, show, save_path)

    def visualize_hierarchy(self, show=True, save_path=None):
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        plot_object = self.topic_model.visualize_hierarchy()
        self._visualize(plot_object, show, save_path)

    def visualize_barchart(self, show=True, save_path=None):
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        plot_object = self.topic_model.visualize_barchart()
        self._visualize(plot_object, show, save_path)

    def visualize_heatmap(self, show=True, save_path=None):
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        plot_object = self.topic_model.visualize_heatmap()
        self._visualize(plot_object, show, save_path)

    def visualize_term_rank(self, show=True, save_path=None):
        # Assuming 'model' was intended to be 'self'
        if self.topic_model is None:
            raise ValueError("Topic model not initialized. Call fit_transform() first.")
        plot_object = self.topic_model.visualize_term_rank()
        self._visualize(plot_object, show, save_path)

# Example usage:
data_path = '/home/marcello/MEGAsync/Promotion/FG/01/01_transcript.txt'
topic_modeling_instance = TopicModeling(data_path)
topic_modeling_instance.fit_transform()

print("get_topic_info: ", topic_modeling_instance.get_topic_info())
print("get_topic: ", topic_modeling_instance.get_topic(0))
print(topic_modeling_instance.get_document_info())

# Other visualizations
topic_modeling_instance.visualize_topics(show=True, save_path='figs/visualize_topics.svg')
topic_modeling_instance.visualize_distribution(topic_modeling_instance.probs[0], show=True,
                                               save_path='figs/visualize_distribution.svg')
topic_modeling_instance.visualize_hierarchy(show=True, save_path='figs/visualize_hierarchy.svg')
topic_modeling_instance.visualize_barchart(show=True, save_path='figs/visualize_barchart.svg')
topic_modeling_instance.visualize_heatmap(show=True, save_path='figs/visualize_heatmap.svg')
topic_modeling_instance.visualize_term_rank(show=True, save_path='figs/visualize_term_rank.svg')

