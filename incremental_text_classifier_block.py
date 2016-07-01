from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nio.block.base import Block
from nio.util.discovery import discoverable
from nio.properties import VersionProperty, ListProperty, StringProperty, \
    PropertyHolder


@input('training')
@input('classify')
@discoverable
class IncrementalTextClassifier(Block):

	version = VersionProperty('0.1.0')

	def __init__(self):
        super().__init__()
        self._classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
        
        self._classifier = Pipeline(
            [('vect', HashingVectorizer()),
             ('tfidf', TfidfTransformer()),
             ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                   n_iter=5, random_state=42))])

    def configure(self, context):
        super().configure(context)

    def process_signals(self, signals, input_id=None):
        """ Process incoming signals.

        This block is a helper, it will just call _process_group and
        notify any signals that get appeneded to the to_notify list.

        Most likely, _process_group will be overridden in subclasses instead
        of this method.
        """
        self.logger.debug(
            "Ready to process {} incoming signals".format(len(signals)))
        signals_to_notify = defaultdict(list)
        with self._safe_lock:
            group_result = self.for_each_group(
                self._process_group, signals, input_id=input_id,
                signals_to_notify=signals_to_notify)
            if group_result:
                signals_to_notify[None] = group_result
        for output_id in signals_to_notify:
            if output_id:
                self.notify_signals(signals_to_notify[output_id],
                                    output_id=output_id)
            else:
                self.notify_signals(signals_to_notify[output_id])

	def _process_group(self, signals, group, input_id, signals_to_notify):
        if input_id == 'setter':
            return self._process_setter_group(signals, group)
        else:
            return self._process_getter_group(signals, group)

    # this method will take the training data point and do a partial fit on the self._classifier
    def _process_training_group(self, signals, group):
    	for signal in signals:
    		self._classifier.partial_fit(signal.data, signal.target)

    # this method notifies the data point passed in as well the classification reached by the classifier
    def _process_classify_group(self, signals, group):
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


