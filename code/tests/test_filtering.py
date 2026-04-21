"""Tests for the smartspim filtering"""

import sys
import unittest
from unittest.mock import patch

import numpy as np
import pywt

sys.path.append("../")
from aind_smartspim_flatfield_estimation import filtering


class SmartspimFiltering(unittest.TestCase):
    """Class for testing smartspim filtering"""

    def test_sigmoid(self):
        """Test the sigmoid function"""
        # Test with scalar
        self.assertAlmostEqual(filtering.sigmoid(np.array(0)), 0.5)
        self.assertAlmostEqual(filtering.sigmoid(np.array(-1)), 1 / (1 + np.exp(1)))
        self.assertAlmostEqual(filtering.sigmoid(np.array(1)), 1 / (1 + np.exp(-1)))

        # Test with array
        data = np.array([-1, 0, 1])
        expected = 1 / (1 + np.exp(-data))
        np.testing.assert_array_almost_equal(filtering.sigmoid(data), expected)

    def test_foreground_fraction(self):
        """Testing foreground fraction"""
        # Test with simple data
        img = np.array([10, 20, 30, 40, 50])
        center = 30
        crossover = 10

        z = (img - center) / crossover
        expected = 1 / (1 + np.exp(-z))
        np.testing.assert_array_almost_equal(
            filtering.foreground_fraction(img, center, crossover), expected
        )

    def test_get_foreground_background_mean(self):
        """Testing get foreground vs background mean"""
        # Test with simple data
        img = np.array([10, 20, 400, 500, 600])
        threshold_mask = 0.3

        # Compute expected results (function now uses float32 internally)
        cell_for = filtering.foreground_fraction(img.astype(np.float32), 400, 20)
        cell_for[cell_for > threshold_mask] = 1
        cell_for[cell_for <= threshold_mask] = 0

        foreground = img[cell_for == 1]
        background = img[cell_for == 0]

        foreground_mean = foreground.mean() if foreground.size else 0.0
        background_mean = background.mean() if background.size else 0.0

        # Call the function
        fg_mean, bg_mean, mask = filtering.get_foreground_background_mean(
            img, threshold_mask
        )

        # Validate results
        self.assertAlmostEqual(fg_mean, foreground_mean)
        self.assertAlmostEqual(bg_mean, background_mean)
        np.testing.assert_array_equal(mask, cell_for)

    def test_empty_image_get_foreground_background_mean(self):
        """
        Testing get foreground vs background
        mean when the image is empty
        """
        # Test with an empty image
        img = np.array([])
        threshold_mask = 0.3

        fg_mean, bg_mean, mask = filtering.get_foreground_background_mean(
            img, threshold_mask
        )

        self.assertEqual(fg_mean, 0.0)
        self.assertEqual(bg_mean, 0.0)
        np.testing.assert_array_equal(mask, img)

    def test_no_foreground(self):
        """
        Testing when there is no foreground
        """
        # Test with all background values
        img = np.array([10, 20, 30, 40, 50])
        threshold_mask = 1.0  # No values will be above this threshold

        fg_mean, bg_mean, mask = filtering.get_foreground_background_mean(
            img, threshold_mask
        )

        self.assertEqual(fg_mean, 0.0)  # No foreground
        self.assertEqual(bg_mean, img.mean())  # All values are background
        np.testing.assert_array_equal(mask, np.zeros_like(img))

    def test_no_background(self):
        """
        Testing with no background in the image
        """
        img = np.array([400, 420, 430, 440, 460])
        threshold_mask = 0.0

        fg_mean, bg_mean, mask = filtering.get_foreground_background_mean(
            img, threshold_mask
        )

        self.assertEqual(fg_mean, img.mean())
        self.assertEqual(bg_mean, 0.0)
        np.testing.assert_array_equal(mask, np.ones_like(img))

    def test_notch(self):
        """testing notch function"""
        # Test with valid inputs
        n = 5
        sigma = 1.0
        result = filtering.notch(n, sigma)
        expected = 1 - np.exp(-(np.arange(n) ** 2) / (2 * sigma**2))
        np.testing.assert_array_almost_equal(result, expected)

        # Test with n = 1 (edge case)
        self.assertAlmostEqual(filtering.notch(1, sigma)[0], 1 - np.exp(0))

        with self.assertRaises(ValueError):
            filtering.notch(0, sigma)  # n <= 0
        with self.assertRaises(ValueError):
            filtering.notch(-1, sigma)  # n <= 0
        with self.assertRaises(ValueError):
            filtering.notch(n, -1)  # sigma <= 0

    def test_gaussian_filter(self):
        """
        Testing gaussian filter
        """
        shape = (3, 5)
        sigma = 1.0
        result = filtering.gaussian_filter(shape, sigma)
        expected_notch = filtering.notch(shape[-1], sigma)
        expected = np.broadcast_to(expected_notch, shape)
        np.testing.assert_array_almost_equal(result, expected)

        # Test with edge case: shape = (1, 1)
        shape = (1, 1)
        result = filtering.gaussian_filter(shape, sigma)
        np.testing.assert_array_equal(result, np.array([[1 - np.exp(0)]]))

    def test_log_space_fft_filtering(self):
        """
        Testing stripe removal with synthetic horizontal stripes
        """
        # Create a synthetic image with horizontal stripes
        input_image = np.tile(np.linspace(1, 100, 100), (100, 1)).astype(np.float32)
        wavelet = "db3"
        level = 1
        sigma = 64
        max_threshold = 4

        # Apply the filter
        result = filtering.log_space_fft_filtering(
            input_image, wavelet, level, sigma, max_threshold
        )

        # Validate the result
        self.assertEqual(result.shape, input_image.shape)
        self.assertTrue(np.all(result > 0))  # Ensure no negative values in the result

    def test_log_space_fft_filtering_small_image(self):
        """
        Testing filtering with a very small image
        """
        # Edge case: small image
        input_image = np.random.rand(4, 4).astype(np.float32)
        result = filtering.log_space_fft_filtering(
            input_image, wavelet="db3", level=1, sigma=64, max_threshold=4
        )
        self.assertEqual(result.shape, input_image.shape)

    def test_normalize_image(self):
        """
        Testing image normalization
        """
        images = [np.array([[1, 2], [3, 4]]), np.array([[0, 5], [10, 15]])]
        normalized = filtering.normalize_image(images)
        self.assertGreaterEqual(normalized.min(), 1.0, "Minimum value should be >= 1.0")
        self.assertLessEqual(normalized.max(), 2.0, "Maximum value should be <= 2.0")
        self.assertEqual(
            normalized.shape, (2, 2, 2), "Output shape should match input list shape"
        )

    def test_invert_image(self):
        """
        Testing image invert
        """
        image = np.array([[0, 1], [2, 3]])
        inverted = filtering.invert_image(image)
        expected = np.array([[3, 2], [1, 0]])
        np.testing.assert_array_equal(
            inverted, expected, "Inverted image values incorrect"
        )

    def test_get_hemisphere_flatfield(self):
        """
        Testing the function tog et the flatfields that
        come from the SmartSPIM microscope. The microscope
        has two lasers and there is one flat per laser.
        """
        tile_config = {"X1": {"Y1": 0, "Y2": 1}, "X2": {"Y1": 0, "Y2": 1}}
        flatfields = [np.array([[1, 1], [1, 1]]), np.array([[2, 2], [2, 2]])]
        flatfield = filtering.get_hemisphere_flatfield(
            "path/to/X1_Y1/test.zarr", tile_config, flatfields
        )
        np.testing.assert_array_equal(
            flatfield, flatfields[0], "Incorrect flatfield returned"
        )

        flatfield = filtering.get_hemisphere_flatfield(
            "path/to/X2_Y2/test.zarr", tile_config, flatfields
        )
        np.testing.assert_array_equal(
            flatfield, flatfields[1], "Incorrect flatfield returned"
        )

        with self.assertRaises(KeyError):
            filtering.get_hemisphere_flatfield(
                "path/to/X3_Y1/test.zarr", tile_config, flatfields
            )

    def test_flatfield_correction(self):
        """
        Testing flatfield correction
        """
        image_tiles = np.array([[[10, 20], [30, 40]]])
        flatfield = np.array([[[2, 2], [2, 2]]])
        darkfield = np.array([[[1, 1], [1, 1]]])
        corrected = filtering.flatfield_correction(image_tiles, flatfield, darkfield)
        expected = np.array([[[4, 9], [14, 19]]], dtype=np.uint16)
        np.testing.assert_array_equal(
            corrected, expected, "Flatfield correction incorrect"
        )

        with self.assertRaises(ValueError):
            filtering.flatfield_correction(image_tiles, flatfield, darkfield[:-1])

    @patch("aind_smartspim_flatfield_estimation.filtering.log_space_fft_filtering")
    @patch(
        "aind_smartspim_flatfield_estimation.filtering.get_foreground_background_mean"
    )
    def test_filter_stripes(
        self, mock_get_foreground_background_mean, mock_log_space_fft_filtering
    ):
        """
        Tests filtering stripes
        """
        image = np.array([[10, 20], [30, 40]])
        no_cells_config = {"wavelet": "db3", "sigma": 64}
        cells_config = {"wavelet": "db3", "sigma": 64}

        mock_get_foreground_background_mean.return_value = (50, 5, None)
        mock_log_space_fft_filtering.return_value = image

        filtered_image = filtering.filter_stripes(
            image,
            "path/to/tile",
            no_cells_config,
            cells_config,
            shadow_correction=None,
        )
        np.testing.assert_array_equal(
            filtered_image, image, "Filtering output mismatch"
        )

        shadow_correction = {
            "retrospective": True,
            "flatfield": np.array([[2, 2], [2, 2]]),
            "darkfield": np.array([[1, 1], [1, 1]]),
            "tile_config": {},
        }
        filtered_image = filtering.filter_stripes(
            image,
            "path/to/tile",
            no_cells_config,
            cells_config,
            shadow_correction=shadow_correction,
        )
        self.assertIsNotNone(filtered_image, "Shadow correction not applied correctly")


class TestForegroundBackgroundFloat32(unittest.TestCase):
    """Verify the float32 fix in get_foreground_background_mean."""

    def test_mask_dtype_not_float16(self):
        img = np.array([10, 20, 400, 500, 600])
        _, _, mask = filtering.get_foreground_background_mean(img, threshold_mask=0.3)
        self.assertNotEqual(mask.dtype, np.float16)

    def test_no_overflow_on_high_intensity(self):
        """Values that would overflow float16 (>65504) must not produce inf."""
        img = np.array([60000, 60001, 60002], dtype=np.float32)
        _, _, mask = filtering.get_foreground_background_mean(img)
        self.assertFalse(np.any(np.isinf(mask)), "sigmoid produced inf with float16")
        self.assertFalse(np.any(np.isnan(mask)))


class TestNormalizeImageFloat32(unittest.TestCase):
    """Verify normalize_image now outputs float32 instead of float16."""

    def test_output_dtype_is_float32(self):
        images = [np.array([[1, 2], [3, 4]]), np.array([[0, 5], [10, 15]])]
        normalized = filtering.normalize_image(images)
        self.assertEqual(normalized.dtype, np.float32)

    def test_range_is_one_to_two(self):
        images = [np.array([[0, 100], [200, 300]])]
        normalized = filtering.normalize_image(images)
        self.assertGreaterEqual(float(normalized.min()), 1.0)
        self.assertLessEqual(float(normalized.max()), 2.0 + 1e-5)

    def test_precision_preserved_vs_float16(self):
        """float32 output must be closer to true result than float16 would be."""
        images = [np.linspace(0, 1, 100)]
        result = filtering.normalize_image(images)
        true_result = 1.0 + (np.array(images) - 0.0) / 1.0
        max_err_f32 = float(np.max(np.abs(result - true_result)))
        max_err_f16 = float(
            np.max(np.abs(true_result.astype(np.float16) - true_result))
        )
        self.assertLess(max_err_f32, max_err_f16 + 1e-6)


class TestFlatfieldCorrectionImportedInFiltering(unittest.TestCase):
    """filtering.flatfield_correction must be the same object as the one
    defined in flatfield_estimation (no duplicate implementation)."""

    def test_same_function_object(self):
        from aind_smartspim_flatfield_estimation import flatfield_estimation

        self.assertIs(
            filtering.flatfield_correction, flatfield_estimation.flatfield_correction
        )

    def test_darkfield_slicing_fix_via_filtering(self):
        """Larger 2-D darkfield must be spatially cropped by the fixed ellipsis slice.

        Uses a single tile (N=1) so expand_dims + crop gives (1, H, W) matching tiles.
        """
        tiles = np.ones((1, 4, 4), dtype=np.float32) * 100
        ff = np.ones((1, 4, 4), dtype=np.float32)
        df_large = np.zeros((8, 8), dtype=np.float32)
        result = filtering.flatfield_correction(tiles, ff, df_large)
        self.assertEqual(result.shape, (1, 4, 4))


class TestGetHemisphereFlatfieldPathFix(unittest.TestCase):
    """Path parsing must work with pathlib.Path objects and non-slash separators."""

    def setUp(self):
        self.tile_config = {"X1": {"Y1": 0, "Y2": 1}}
        self.flatfields = [np.zeros((2, 2)), np.ones((2, 2))]

    def test_string_path(self):
        ff = filtering.get_hemisphere_flatfield(
            "some/path/X1_Y1/tile.zarr", self.tile_config, self.flatfields
        )
        np.testing.assert_array_equal(ff, self.flatfields[0])

    def test_pathlib_path_object(self):
        from pathlib import Path

        ff = filtering.get_hemisphere_flatfield(
            Path("some/path/X1_Y1/tile.zarr"), self.tile_config, self.flatfields
        )
        np.testing.assert_array_equal(ff, self.flatfields[0])

    def test_missing_x_raises_key_error(self):
        with self.assertRaises(KeyError):
            filtering.get_hemisphere_flatfield(
                "some/path/X9_Y1/tile.zarr", self.tile_config, self.flatfields
            )

    def test_missing_y_raises_key_error(self):
        with self.assertRaises(KeyError):
            filtering.get_hemisphere_flatfield(
                "some/path/X1_Y9/tile.zarr", self.tile_config, self.flatfields
            )


class TestChPowerSimplification(unittest.TestCase):
    """np.abs(ch) and np.sqrt(ch**2) must produce identical results."""

    def test_abs_equals_sqrt_sq(self):
        rng = np.random.default_rng(0)
        ch = rng.standard_normal((16, 32))
        np.testing.assert_array_almost_equal(np.abs(ch), np.sqrt(ch**2))


class TestLogSpaceFftFilteringInverseLog(unittest.TestCase):
    """Issue #1: inverse of log(1+x) must be exp(y)-1, not exp(y)+1."""

    def test_identity_on_stripe_free_image(self):
        """A perfectly uniform image should pass through the filter close to unchanged."""
        input_image = np.full((64, 64), 50.0, dtype=np.float32)
        result = filtering.log_space_fft_filtering(
            input_image, wavelet="db3", level=1, sigma=64, max_threshold=4
        )
        self.assertEqual(result.shape, input_image.shape)
        # With the correct inverse (exp-1), values should stay near 50, not near 52
        mean_error = float(np.abs(result - input_image).mean())
        self.assertLess(mean_error, 5.0, "Mean deviation too large — check inverse log")

    def test_output_not_shifted_by_two(self):
        """With the wrong +1 the output would be ~2 higher than input for flat images."""
        input_image = np.ones((32, 32), dtype=np.float32) * 10.0
        result = filtering.log_space_fft_filtering(
            input_image, wavelet="db3", level=1, sigma=32, max_threshold=4
        )
        mean_val = float(result.mean())
        # Correct: ~10. Wrong (+1 bug): ~12.
        self.assertLess(
            abs(mean_val - 10.0), 2.0, f"Output mean {mean_val:.2f} suggests +1 bug"
        )


class TestNormalizeImageConstantInput(unittest.TestCase):
    """Issue #9: normalize_image must not return NaN/inf for constant images."""

    def test_constant_image_returns_ones(self):
        images = [np.full((4, 4), 42.0)]
        result = filtering.normalize_image(images)
        self.assertFalse(np.any(np.isnan(result)), "NaN found in constant-image result")
        self.assertFalse(np.any(np.isinf(result)), "inf found in constant-image result")
        np.testing.assert_array_equal(result, np.ones_like(result))

    def test_constant_image_dtype_is_float32(self):
        images = [np.zeros((3, 3))]
        result = filtering.normalize_image(images)
        self.assertEqual(result.dtype, np.float32)

    def test_normal_image_still_works(self):
        images = [np.array([[0.0, 1.0], [2.0, 3.0]])]
        result = filtering.normalize_image(images)
        self.assertAlmostEqual(float(result.min()), 1.0, places=5)
        self.assertAlmostEqual(float(result.max()), 2.0, places=5)
