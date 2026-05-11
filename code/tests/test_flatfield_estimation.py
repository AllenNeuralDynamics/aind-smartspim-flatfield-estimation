"""Tests for flatfield estimation functions."""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# basicpy pulls in jax/jaxlib at import time, which may not be available on
# every machine (e.g. jaxlib built for AVX on a non-AVX CPU).  Stub the
# package out before the module under test is loaded so that the rest of
# flatfield_estimation can be imported and tested normally.
sys.modules.setdefault("basicpy", MagicMock())

from aind_smartspim_flatfield_estimation.flatfield_estimation import (  # noqa: E402
    create_median_flatfield,
    estimate_flats_per_laser,
    flatfield_correction,
    shading_correction,
    unify_fields,
)


class TestShadingCorrection(unittest.TestCase):
    @patch("aind_smartspim_flatfield_estimation.flatfield_estimation.BaSiC")
    def test_shading_correction_returns_expected_keys(self, mock_basic):
        mock_obj = MagicMock()
        mock_obj.flatfield = np.ones((5, 5))
        mock_obj.darkfield = np.zeros((5, 5))
        mock_obj.baseline = np.zeros((5,))
        mock_basic.return_value = mock_obj

        slides = [np.random.rand(5, 5) for _ in range(3)]
        result = shading_correction(slides, {})

        self.assertIn("flatfield", result)
        self.assertIn("darkfield", result)
        self.assertIn("baseline", result)
        np.testing.assert_array_equal(result["flatfield"], mock_obj.flatfield)
        np.testing.assert_array_equal(result["darkfield"], mock_obj.darkfield)

    @patch("aind_smartspim_flatfield_estimation.flatfield_estimation.BaSiC")
    def test_shading_correction_no_dead_list(self, mock_basic):
        """shading_results = [] dead code is gone; result must be a dict."""
        mock_obj = MagicMock()
        mock_obj.flatfield = np.ones((4, 4))
        mock_obj.darkfield = np.zeros((4, 4))
        mock_obj.baseline = np.zeros((4,))
        mock_basic.return_value = mock_obj

        result = shading_correction([np.random.rand(4, 4)], {})
        self.assertIsInstance(result, dict)


class TestFlatfieldCorrection(unittest.TestCase):
    def _make_inputs(self, n=1, h=4, w=4):
        tiles = (np.random.randint(100, 500, (n, h, w))).astype(np.float32)
        flatfield = np.ones((n, h, w), dtype=np.float32) * 2.0
        darkfield = np.zeros((n, h, w), dtype=np.float32)
        return tiles, flatfield, darkfield

    def test_basic_correction_shape_and_dtype(self):
        tiles, ff, df = self._make_inputs(n=3, h=4, w=4)
        result = flatfield_correction(tiles, ff, df)
        self.assertEqual(result.shape, (3, 4, 4))
        self.assertEqual(result.dtype, np.uint16)

    def test_known_values(self):
        tiles = np.array([[[10, 20], [30, 40]]], dtype=np.float32)
        ff = np.array([[[2, 2], [2, 2]]], dtype=np.float32)
        df = np.array([[[1, 1], [1, 1]]], dtype=np.float32)
        result = flatfield_correction(tiles, ff, df)
        expected = np.array([[[4, 9], [14, 19]]], dtype=np.uint16)
        np.testing.assert_array_equal(result, expected)

    def test_2d_flatfield_expanded_correctly(self):
        """2-D flatfield/darkfield should auto-expand to match 3-D tiles."""
        tiles = np.array([[[10, 20], [30, 40]]], dtype=np.float32)
        ff = np.array([[2, 2], [2, 2]], dtype=np.float32)
        df = np.array([[1, 1], [1, 1]], dtype=np.float32)
        result = flatfield_correction(tiles, ff, df)
        expected = np.array([[[4, 9], [14, 19]]], dtype=np.uint16)
        np.testing.assert_array_equal(result, expected)

    def test_darkfield_slicing_uses_ellipsis(self):
        """Larger 2-D darkfield must be cropped along the correct spatial axes.

        The old code used darkfield[:H, :W] which, after expand_dims makes the
        array 3-D (1, H_d, W_d), would crop dim-0 and dim-1 instead of the
        spatial dims.  The ellipsis fix darkfield[..., :H, :W] correctly crops
        the last two axes regardless of how many batch dims precede them.
        """
        h, w = 4, 4
        # Single tile so expand_dims gives (1, H_d, W_d) → crop → (1, H, W)
        tiles = np.ones((1, h, w), dtype=np.float32) * 100
        ff = np.ones((1, h, w), dtype=np.float32)
        # Darkfield is 2-D and larger; after expand+crop it should be (1, 4, 4)
        df_larger = np.zeros((h + 4, w + 4), dtype=np.float32)
        result = flatfield_correction(tiles, ff, df_larger)
        self.assertEqual(result.shape, (1, h, w))

    def test_darkfield_3d_crop_correct_axes(self):
        """3-D darkfield larger than tiles must be cropped on the last two axes."""
        n, h, w = 2, 4, 4
        tiles = np.ones((n, h, w), dtype=np.float32) * 50
        ff = np.ones((n, h, w), dtype=np.float32)
        # Darkfield already 3-D (same N) but spatially larger
        df_3d = np.zeros((n, h + 4, w + 4), dtype=np.float32)
        result = flatfield_correction(tiles, ff, df_3d)
        self.assertEqual(result.shape, (n, h, w))

    def test_output_clipped_to_uint16(self):
        tiles = np.array([[[65535, 65535]]], dtype=np.float32)
        ff = np.array([[[0.5, 0.5]]], dtype=np.float32)
        df = np.zeros((1, 1, 2), dtype=np.float32)
        result = flatfield_correction(tiles, ff, df)
        self.assertTrue(np.all(result <= 65535))
        self.assertEqual(result.dtype, np.uint16)

    def test_zero_flatfield_does_not_produce_inf(self):
        """Pixels where flatfield == 0 must produce 0, not inf or NaN."""
        tiles = np.array([[[100, 200], [300, 400]]], dtype=np.float32)
        ff = np.array([[[0, 1], [1, 0]]], dtype=np.float32)
        df = np.zeros((1, 2, 2), dtype=np.float32)
        result = flatfield_correction(tiles, ff, df)
        self.assertFalse(np.any(np.isinf(result.astype(np.float32))))
        self.assertFalse(np.any(np.isnan(result.astype(np.float32))))
        self.assertEqual(result[0, 0, 0], 0)
        self.assertEqual(result[0, 1, 1], 0)

    def test_zero_flatfield_preserves_nonzero_pixels(self):
        """Pixels where flatfield > 0 must still be corrected normally."""
        tiles = np.array([[[0, 200]]], dtype=np.float32)
        ff = np.array([[[0, 2]]], dtype=np.float32)
        df = np.zeros((1, 1, 2), dtype=np.float32)
        result = flatfield_correction(tiles, ff, df)
        self.assertEqual(result[0, 0, 1], 100)

    def test_darkfield_shape_mismatch_raises(self):
        tiles = np.array([[[10, 20], [30, 40]]])
        ff = np.array([[[2, 2], [2, 2]]])
        df_wrong = np.array([[[1]]])
        with self.assertRaises(ValueError):
            flatfield_correction(tiles, ff, df_wrong)

    def test_flatfield_shape_mismatch_raises(self):
        tiles = np.array([[[10, 20], [30, 40]]])
        ff_wrong = np.array([[[2]]])
        df = np.array([[[1, 1], [1, 1]]])
        with self.assertRaises(ValueError):
            flatfield_correction(tiles, ff_wrong, df)

    def test_baseline_zeros_when_none(self):
        tiles = np.array([[[10, 20], [30, 40]]], dtype=np.float32)
        ff = np.array([[[1, 1], [1, 1]]], dtype=np.float32)
        df = np.zeros((1, 2, 2), dtype=np.float32)
        result_no_baseline = flatfield_correction(tiles, ff, df, baseline=None)
        result_zero_baseline = flatfield_correction(tiles, ff, df, baseline=np.zeros(1))
        np.testing.assert_array_equal(result_no_baseline, result_zero_baseline)


class TestCreateMedianFlatfield(unittest.TestCase):
    def test_output_shape_matches_input(self):
        ff = np.random.rand(10, 10)
        result = create_median_flatfield(ff, smooth=False)
        self.assertEqual(result.shape, ff.shape)

    def test_smooth_true_returns_same_shape(self):
        ff = np.random.rand(20, 20)
        result = create_median_flatfield(ff, smooth=True)
        self.assertEqual(result.shape, ff.shape)

    def test_median_row_is_tiled(self):
        ff = np.ones((5, 5)) * np.arange(1, 6)[:, np.newaxis]
        result = create_median_flatfield(ff, smooth=False)
        for col in range(ff.shape[1]):
            np.testing.assert_array_almost_equal(result[:, col], result[:, 0])


class TestEstimateFlatsPerLaser(unittest.TestCase):
    @patch(
        "aind_smartspim_flatfield_estimation.flatfield_estimation.shading_correction"
    )
    def test_keys_match_tiles_per_side(self, mock_sc):
        mock_sc.return_value = {
            "flatfield": np.ones((5, 5)),
            "darkfield": np.zeros((5, 5)),
            "baseline": np.zeros((5,)),
        }
        tiles = {"left": [np.random.rand(5, 5)], "right": [np.random.rand(5, 5)]}
        result = estimate_flats_per_laser(tiles, {})
        self.assertSetEqual(set(result.keys()), {"left", "right"})
        self.assertEqual(mock_sc.call_count, 2)


class TestUnifyFields(unittest.TestCase):
    def _make_fields(self, n=3, shape=(5, 5)):
        flatfields = [np.random.rand(*shape) for _ in range(n)]
        darkfields = [np.random.rand(*shape) for _ in range(n)]
        baselines = [np.random.rand(shape[0]) for _ in range(n)]
        return flatfields, darkfields, baselines

    def test_median_output_shape(self):
        ff, df, bl = self._make_fields()
        flatfield, darkfield, baseline = unify_fields(ff, df, bl, mode="median")
        self.assertEqual(flatfield.shape, (5, 5))
        self.assertEqual(darkfield.shape, (5, 5))
        self.assertEqual(baseline.shape, (5,))

    def test_output_dtype_is_float32_not_float16(self):
        ff, df, bl = self._make_fields()
        flatfield, darkfield, baseline = unify_fields(ff, df, bl, mode="median")
        self.assertEqual(flatfield.dtype, np.float32)
        self.assertEqual(darkfield.dtype, np.float32)
        self.assertEqual(baseline.dtype, np.float32)

    def test_mean_mode(self):
        ff = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        df = [np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])]
        bl = [np.array([0.0]), np.array([0.0])]
        flatfield, _, _ = unify_fields(ff, df, bl, mode="mean")
        np.testing.assert_array_almost_equal(flatfield, [[2.0, 3.0]], decimal=4)

    def test_mip_mode(self):
        ff = [np.array([[1.0, 4.0]]), np.array([[3.0, 2.0]])]
        df = [np.array([[1.0, 2.0]]), np.array([[3.0, 0.0]])]
        bl = [np.array([1.0]), np.array([5.0])]
        flatfield, darkfield, baseline = unify_fields(ff, df, bl, mode="mip")
        np.testing.assert_array_almost_equal(flatfield, [[3.0, 4.0]], decimal=4)
        np.testing.assert_array_almost_equal(darkfield, [[1.0, 0.0]], decimal=4)

    def test_invalid_mode_raises(self):
        ff, df, bl = self._make_fields()
        with self.assertRaises(NotImplementedError):
            unify_fields(ff, df, bl, mode="invalid")


if __name__ == "__main__":
    unittest.main()
