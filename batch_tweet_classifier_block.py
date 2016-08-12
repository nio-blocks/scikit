from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nio.block.base import Block
from nio.signal.base import Signal
from nio.util.discovery import discoverable
from nio.properties import VersionProperty, ListProperty, StringProperty, \
    PropertyHolder
from nio.block.terminals import input
from nio.block import output
import re

@input('classify')
@input('training')
@output('ready', label='Ready')
@output('result', label='Result')
@discoverable
class BatchTweetClassifier(Block):
    version = VersionProperty('0.1.0')

    def __init__(self):
        super().__init__()
        self._train_complete = False
        self._classifier = Pipeline(
            [('vect', CountVectorizer()),
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
        self._process_group(signals, input_id)


    def _process_group(self, signals, input_id):
        if input_id == 'training':
            return self._process_training_group(signals)
        else:
            return self._process_classify_group(signals)

    def _process_training_group(self, signals):


        if self._train_complete == True:
            #Do not train because the classifier has already been built
            self.logger.warning("Classifier has already been trained")
        else:
            self._data = {}
            self._data['data'] = []
            self._data['target'] = []
            for signal in signals:
                # each time there is a new label, add it to target_names
                self._data['data'].append(signal.data)
                self._data['target'].append(signal.target)
      
            self._data['data'] = [self.processTweet(t) for t in self._data['data']]
            self._classifier.fit(self._data['data'], self._data['target'])
            self._train_complete = True
            self.notify_signals([Signal({'ready': True})], 'ready')

    def _process_classify_group(self, signals):
        predicted_signals = []
        self.logger.debug("Value of _classifier : {}".format(self._classifier))
        for signal in signals:
            try:
                signal.data = self.processTweet(signal.data)
                predicted = self._classifier.predict([signal.data])
                self.logger.debug("Value of predicted : {}".format(predicted))
                signal.label = predicted[0]
                predicted_signals.append(signal)
            except:
                self.logger.warning("Classifier does not have training data",
                                    exc_info=True)
        self.notify_signals(predicted_signals, 'result')


    def processTweet(self, tweet):
        # process the tweets

        #Convert to lower case
        tweet = tweet.lower()
        #Convert www.* or https?://* to URL
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
        #Convert @username to AT_USER
        tweet = re.sub('@[^\s]+','AT_USER',tweet)
        #Remove additional white spaces
        tweet = re.sub('[\s]+', ' ', tweet)
        #Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        #trim
        tweet = tweet.strip('\'"')
        return tweet








