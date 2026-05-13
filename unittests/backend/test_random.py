import unittest

import numpy

from onnx_light.backend.random import rand, randint, randn


class TestCustomRandom(unittest.TestCase):
    def test_rand_is_deterministic_with_seed(self):
        first = rand(2, 3, seed=42)
        second = rand(2, 3, seed=42)
        numpy.testing.assert_array_equal(first, second)

    def test_rand_and_randn_default_seed_are_deterministic(self):
        numpy.testing.assert_array_equal(rand(4), rand(4))
        numpy.testing.assert_array_equal(randn(4), randn(4))

    def test_randint_is_deterministic_with_seed(self):
        first = randint(0, 10, size=(2, 4), seed=5)
        second = randint(0, 10, size=(2, 4), seed=5)
        numpy.testing.assert_array_equal(first, second)

    def test_rand_and_randn_expected_values(self):
        expected_rand = numpy.array(
            [0.3898297483912715, 0.01678829452815611, 0.9007606806068834], dtype=numpy.float64
        )
        expected_randn = numpy.array(
            [-1.0009541026316917, 1.3335403660869787, 0.5188620948808316], dtype=numpy.float64
        )
        numpy.testing.assert_allclose(rand(3, seed=7), expected_rand, rtol=0.0, atol=0.0)
        numpy.testing.assert_allclose(randn(3, seed=7), expected_randn, rtol=0.0, atol=0.0)

    def test_randint_expected_values(self):
        expected = numpy.array([2, 3, 0, 3, 5], dtype=numpy.int64)
        numpy.testing.assert_array_equal(randint(0, 7, size=5, seed=7), expected)

    def test_randint_supports_numpy_style_high_only(self):
        values = randint(5, size=100, seed=12)
        self.assertTrue(((values >= 0) & (values < 5)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
