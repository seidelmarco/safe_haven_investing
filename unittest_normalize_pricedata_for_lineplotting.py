import unittest


def normalize_pricedata(a: int, b: int) -> int:
    """

    :param a:
    :param b:
    :return:
    """
    return a + b


class TestNormalizer(unittest.TestCase):
    def test_normalize_pricedata(self):
        self.assertEqual(add(1, 2), 3)


TestNormalizer()