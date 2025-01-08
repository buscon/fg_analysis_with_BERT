from typing import List, Union, Tuple
import numpy as np
import logging
from bertopic import BERTopic


class LoggerToFile(logging.Logger):
    def __init__(self, name, level=logging.NOTSET, log_to_file=False,
                 filename="bertopic.log"):

        super().__init__(name, level)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.addHandler(console_handler)

        # File Handler (optional)
        if log_to_file:
            file_handler = logging.FileHandler(filename)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.addHandler(file_handler)

class BERTopicModified(BERTopic):
    def __init__(self, *args, log_to_file=False, log_file_path="bertopic.log",
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.logger = LoggerToFile("BERTopic", log_to_file=log_to_file,
                                   filename=log_file_path)
        self.logger.info(f"Initialized BERTopic with parameters: {args}, {kwargs}")

    # Example method where you log results
    def fit_transform(self, documents: List[str], embeddings: np.ndarray =
                      None, images: List[str] = None, y: Union[List[int],
                                                               np.ndarray] =
                      None) -> Tuple[List[int], Union[np.ndarray, None]]:

        self.logger.info("Starting fit_transform method")

        predictions, probabilities = super().fit_transform(documents,
                                                           embeddings, images,
                                                           y)

        # Logging the end of the method and results
        self.logger.info("Completed fit_transform method")
        self.logger.info(f"Topics: {predictions}")
        self.logger.info(f"Topics: {self.get_topic_info()}")
        self.logger.info(f"Topic Names: {self.get_topic_info().Name}")
        self.logger.info(f"Probabilities: {self.probabilities_}")

        return predictions, self.probabilities_

# Usage
# topic_model = BERTopicModified(log_to_file=True,
                         # log_file_path="my_log_file.log")

