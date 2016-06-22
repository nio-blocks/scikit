from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nio.block.base import Block
from nio.util.discovery import discoverable
from nio.properties import VersionProperty, ListProperty, StringProperty, \
    PropertyHolder


class TrainingSetDataPoint(PropertyHolder):

    data = StringProperty(title="Data")
    target = StringProperty(title="Target")


@discoverable
class TextClassifier(Block):

    training_set = ListProperty(
        TrainingSetDataPoint, title="Training Set", default=[])
    version = VersionProperty('0.1.0')

    def __init__(self):
        super().__init__()
        self._classifier = Pipeline(
            [('vect', CountVectorizer()),
             ('tfidf', TfidfTransformer()),
             ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                   n_iter=5, random_state=42))])

    def configure(self, context):
        super().configure(context)
        try:
            training_data = []
            training_targets = []
            for data_point in self.training_set():
                training_data.append(data_point.data())
                training_targets.append(data_point.target())
            self._classifier.fit(training_data, training_targets)
        except:
            self.logger.warning("No training data available during configure")

    def process_signals(self, signals):
        predicted_signals = []
        for signal in signals:
            try:
                predicted = self._classifier.predict([signal.sample])
                signal.target = predicted[0]
                predicted_signals.append(signal)
            except:
                self.logger.warning("Classifier does not have training data",
                                    exc_info=True)
        self.notify_signals(predicted_signals)
