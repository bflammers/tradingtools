
import unittest
import tradingtools.utils as utils


class TestSplitPair(unittest.TestCase):

    def test_slash(self):

        base, quote = utils.split_pair("AAA/BBB")
        self.assertEqual(base, "AAA")
        self.assertEqual(quote, "BBB")

    def test_dash(self):
        base, quote = utils.split_pair("AAA-BBB")
        self.assertEqual(base, "AAA")
        self.assertEqual(quote, "BBB")

    def test_error(self):
        self.assertRaises(Exception, lambda: utils.split_pair("AAA.BBB"))

