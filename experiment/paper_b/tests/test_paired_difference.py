from __future__ import annotations

import unittest

import numpy as np

from experiment.paper_b.paired_difference import (
    encode_context_difference,
    encode_paired_difference,
)


class PairedDifferenceTest(unittest.TestCase):
    def test_identical_pair_is_zero(self) -> None:
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        encoded = encode_paired_difference(image, image)
        self.assertEqual(encoded.shape, (8, 8, 3))
        self.assertEqual(int(encoded.max()), 0)

    def test_dark_and_bright_changes_use_separate_channels(self) -> None:
        reference = np.full((16, 16), 128, dtype=np.uint8)
        candidate = reference.copy()
        candidate[4:8, 4:8] = 80
        candidate[10:14, 10:14] = 180
        encoded = encode_paired_difference(candidate, reference, noise_floor=0.0)
        self.assertGreater(int(encoded[5, 5, 0]), int(encoded[5, 5, 1]))
        self.assertGreater(int(encoded[11, 11, 1]), int(encoded[11, 11, 0]))
        self.assertGreater(int(encoded[:, :, 2].max()), 0)

    def test_rejects_invalid_pair(self) -> None:
        with self.assertRaises(ValueError):
            encode_paired_difference(np.zeros((4, 4)), np.zeros((5, 5)))

    def test_context_encoding_preserves_unchanged_candidate(self) -> None:
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        encoded = encode_context_difference(image, image)
        for channel in range(3):
            np.testing.assert_array_equal(encoded[:, :, channel], image)

    def test_context_encoding_marks_change_without_removing_context(self) -> None:
        reference = np.full((16, 16), 128, dtype=np.uint8)
        candidate = reference.copy()
        candidate[4:12, 4:12] = 80
        encoded = encode_context_difference(candidate, reference, noise_floor=0.0)
        np.testing.assert_array_equal(encoded[:, :, 0], candidate)
        self.assertNotEqual(int(encoded[7, 7, 1]), int(encoded[7, 7, 2]))


if __name__ == "__main__":
    unittest.main()
