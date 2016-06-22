from unittest.mock import MagicMock
from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..text_classifier_block import TextClassifier


class TestTextClassifier(NIOBlockTestCase):

    def test_process_sample_data_signal_with_no_fit(self):
        """Sample data is not predicted if no training data."""
        blk = TextClassifier()
        self.configure_block(blk, {})
        blk.logger = MagicMock()
        blk.start()
        blk.process_signals([Signal({"sample": "this is sample text"})])
        blk.stop()
        self.assert_num_signals_notified(0)
        self.assertEqual(blk.logger.warning.call_count, 1)

    def test_process_sample_data_signals_with_no_fit(self):
        """Sample data is not predicted if no training data."""
        blk = TextClassifier()
        self.configure_block(blk, {})
        blk.logger = MagicMock()
        blk.start()
        blk.process_signals([Signal({"sample": "this is sample text"}),
                             Signal({"sample": "this is more text"})])
        blk.stop()
        self.assert_num_signals_notified(0)
        self.assertEqual(blk.logger.warning.call_count, 2)

    def test_process_sample_data_signal(self):
        """Sample data makes a prediction and gets the target value."""
        blk = TextClassifier()
        self.configure_block(blk, {
            "training_set": [
                {"data": "this is sample text", "target": "sample"},
                {"data": "this is more text", "target": "more"},
            ]
        })
        blk.start()
        blk.process_signals([Signal({"sample": "this is sample text"})])
        blk.stop()
        self.assert_num_signals_notified(1)
        self.assertTrue(
            'target' in self.last_notified[DEFAULT_TERMINAL][0].to_dict())
        self.assertEqual(
            self.last_notified[DEFAULT_TERMINAL][0].to_dict()['target'],
            "sample")

    def test_process_sample_data_signals(self):
        """Sample data makes a prediction and gets the target value."""
        blk = TextClassifier()
        self.configure_block(blk, {
            "training_set": [
                {"data": "this is sample text", "target": "sample"},
                {"data": "this is more text", "target": "more"},
            ]
        })
        blk.start()
        blk.process_signals([Signal({"sample": "this is sample text"}),
                             Signal({"sample": "this is more text"})])
        blk.stop()
        self.assert_num_signals_notified(2)
        self.assertTrue(
            'target' in self.last_notified[DEFAULT_TERMINAL][0].to_dict())
        self.assertTrue(
            'target' in self.last_notified[DEFAULT_TERMINAL][1].to_dict())
        self.assertEqual(
            self.last_notified[DEFAULT_TERMINAL][0].to_dict()['target'],
            "sample")
        self.assertEqual(
            self.last_notified[DEFAULT_TERMINAL][1].to_dict()['target'],
            "more")
