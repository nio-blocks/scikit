from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nio.block.base import Block
from nio.signal.base import Signal
from nio.util.discovery import discoverable
from nio.properties import VersionProperty
from nio.block.terminals import input
from nio.block import output
import re


@input('classify')
@input('training')
@output('ready', label='Ready')
@output('result', label='Result')
@discoverable
class BatchTextClassifier(Block):

    version = VersionProperty('0.1.0')

    def __init__(self):
        super().__init__()
        self._train_complete = False
        self._classifier = Pipeline(
            [('vect', CountVectorizer()),
             ('tfidf', TfidfTransformer()),
             ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                   n_iter=5, random_state=42))])

    def process_signals(self, signals, input_id=None):
        """ Process incoming signals.

        This block is a helper, it will just call _process_group and
        notify any signals that get appeneded to the to_notify list.

        Most likely, _process_group will be overridden in subclasses instead
        of this method.
        """
        self.logger.debug(
            "Ready to process {} incoming signals".format(len(signals)))
        self._process_group(signals, input_id)

    def _process_group(self, signals, input_id):
        if input_id == 'training':
            return self._process_training_group(signals)
        else:
            return self._process_classify_group(signals)

    def _process_training_group(self, signals):
        if self._train_complete is True:
            # Do not train because the classifier has already been built
            self.logger.warning("Classifier has already been trained")
        else:
            data = {}
            data['data'] = []
            data['target'] = []
            for signal in signals:
                # each time there is a new label, add it to target_names
                data['data'].append(signal.data)
                data['target'].append(signal.target)
            self._classifier.fit(data['data'], data['target'])
            self._train_complete = True
            self.notify_signals([Signal({'ready': True})], 'ready')

    def _process_classify_group(self, signals):
        predicted_signals = []
        self.logger.debug("Value of _classifier : {}".format(self._classifier))
        for signal in signals:
            try:
                predicted = self._classifier.predict([signal.data])
                self.logger.debug("Value of predicted : {}".format(predicted))
                signal.label = predicted[0]
                predicted_signals.append(signal)
            except:
                self.logger.warning("Classifier does not have training data",
                                    exc_info=True)
        self.notify_signals(predicted_signals, 'result')
