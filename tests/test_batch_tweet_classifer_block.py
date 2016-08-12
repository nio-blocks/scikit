from unittest.mock import MagicMock
from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..batch_tweet_classifier_block import BatchTweetClassifier

class TestBatchTweetClassifier(NIOBlockTestCase):

    def test_process_data(self):
        blk = BatchTweetClassifier()
        self.configure_block(blk, {})
        blk.logger = MagicMock()
        blk.start()
        blk.process_signals([Signal({
                                    "text": "this is sample text", 
                                    "prof_img": "img.jpeg", 
                                    "target": "sample_target"}),
                            Signal({
                                    "text": "this text is different from the previous text", 
                                    "prof_img": "img2.jpeg",
                                    "target": "different_target"})], input_id='training')
       

        blk.process_signals([Signal({
                                    "text": "this is sample text", 
                                    "prof_img": "img.jpeg"})], input_id='classify')
        blk.stop()
        self.assert_num_signals_notified(1)
        self.assertTrue(
            'sample_target' == self.last_notified[DEFAULT_TERMINAL][0].to_dict()['target'])