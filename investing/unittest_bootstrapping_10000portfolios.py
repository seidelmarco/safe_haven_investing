import unittest


def bootstrap_portfolios(a: int, b: int) -> int:
    """

    :param a:
    :param b:
    :return:
    """
    return a + b


class TestBootstrapper(unittest.TestCase):
    def test_bootstrap_portfolios(self):
        self.assertEqual(add(1, 2), 3)


TestBootstrapper()