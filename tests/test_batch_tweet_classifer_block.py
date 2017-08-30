from unittest.mock import MagicMock

from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase

from ..batch_text_classifier_block import BatchTextClassifier


class TestBatchTweetClassifier(NIOBlockTestCase):

    def test_process_data(self):
        blk = BatchTextClassifier()
        self.configure_block(blk, {})
        blk.logger = MagicMock()
        blk.start()
        blk.process_signals([
            Signal({
                "data": "this is sample text",
                "prof_img": "img.jpeg",
                "target": "sample_target",
            }),
            Signal({
                "data": "this text is different from the previous text",
                "prof_img": "img2.jpeg",
                "target": "different_target",
            })
        ], input_id='training')
        blk.process_signals([
            Signal({
                "data": "this is sample text",
                "prof_img": "img.jpeg",
            })
        ], input_id='classify')
        blk.stop()
        self.assert_num_signals_notified(2)
        self.assertDictEqual(self.last_notified["ready"][0].to_dict(), {
            "ready": True,
        })
        self.assertDictEqual(self.last_notified["result"][0].to_dict(), {
            "data": "this is sample text",
            "prof_img": "img.jpeg",
            "label": "sample_target",
        })
