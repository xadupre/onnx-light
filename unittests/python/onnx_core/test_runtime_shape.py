"""Tests the concrete native Shape exposed through the core bindings."""

import copy
import unittest

from onnx_light.onnx_core.shape_inference import Shape


class TestRuntimeShape(unittest.TestCase):
    def test_scalar_and_dimensions(self):
        scalar = Shape()
        self.assertEqual(len(scalar), 0)
        self.assertTrue(scalar.empty())
        self.assertEqual(scalar.product(), 1)
        for dims in ([2, 3], (2, 3)):
            shape = Shape(dims)
            self.assertEqual(list(shape), [2, 3])
            self.assertEqual(shape.dims(), [2, 3])
            self.assertEqual(shape.rank(), 2)
            self.assertEqual(shape.product(), 6)
            self.assertEqual(repr(shape), "Shape([2, 3])")
            self.assertEqual(shape, Shape([2, 3]))
            self.assertNotEqual(shape, Shape([3, 2]))
            self.assertEqual(shape[-1], 3)
            shape[-2] = 4
            self.assertEqual(shape[0], 4)
            shape.append(5)
            self.assertEqual(list(shape), [4, 3, 5])

    def test_positional_dimensions(self):
        for dims in ((), (3,), (2, 3), (2, 0, 4), tuple(range(16))):
            with self.subTest(dims=dims):
                self.assertEqual(Shape(*dims), Shape(dims))
        self.assertEqual(list(Shape(2, 3)), [2, 3])
        self.assertEqual(list(Shape(3)), [3])
        self.assertEqual(Shape(dims=[2, 3]), Shape(2, 3))
        self.assertEqual(Shape(shape=Shape(2, 3)), Shape(2, 3))
        with self.assertRaisesRegex(ValueError, "maximum"):
            Shape(*([1] * 17))
        for dims in ((1.5,), ("N",), (1 << 63,), (2, 3.5), ([2], 3), (None,)):
            with self.subTest(dims=dims), self.assertRaises(TypeError):
                Shape(*dims)

    def test_independent_copies_and_iteration(self):
        shape = Shape([2, 3])
        copies = (Shape(shape), copy.copy(shape))
        dims = shape.dims()
        iterator = iter(shape)
        shape[0] = 9
        dims[0] = 8
        self.assertEqual(list(iterator), [2, 3])
        for copied in copies:
            self.assertEqual(list(copied), [2, 3])
        self.assertEqual(list(shape), [9, 3])

    def test_invalid_indices_rank_and_values(self):
        shape = Shape([2, 3])
        for index in (-3, 2):
            with self.assertRaises(IndexError):
                shape[index]
            with self.assertRaises(IndexError):
                shape[index] = 4
        with self.assertRaises(IndexError):
            Shape()[0]
        for dims in ([1.5], ["N"], [1 << 63]):
            with self.assertRaises(TypeError):
                Shape(dims)
        self.assertEqual(Shape.max_rank, 16)
        with self.assertRaisesRegex(ValueError, "maximum"):
            Shape([1] * 17)
        full = Shape([1] * 16)
        with self.assertRaisesRegex(ValueError, "maximum"):
            full.append(1)
        self.assertEqual(len(full), 16)
        with self.assertRaisesRegex(ValueError, "negative"):
            Shape([-1, 2]).product()
        with self.assertRaisesRegex(ValueError, "overflow"):
            Shape([1 << 62, 4]).product()


if __name__ == "__main__":
    unittest.main()
