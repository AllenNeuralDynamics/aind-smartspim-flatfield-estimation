"""Tests for the smartspim filtering"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aind_smartspim_flatfield_estimation import filtering


class SmartspimFiltering(unittest.TestCase):
    """Class for testing smartspim filtering"""

    @classmethod
    def setUpClass(cls):
        """Setup basic setting accross tests"""
        cls.temp_folder = tempfile.mkdtemp(prefix="unittest_")

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

        # Compute expected results
        cell_for = filtering.foreground_fraction(img.astype(np.float16), 400, 20)
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
        # Test with all foreground values
        img = np.array([400, 420, 430, 440, 460])
        threshold_mask = 0.0  # All values will be above this threshold

        fg_mean, bg_mean, mask = filtering.get_foreground_background_mean(
            img, threshold_mask
        )

        self.assertEqual(fg_mean, img.mean())  # All values are foreground
        self.assertEqual(bg_mean, 0.0)  # No background
        np.testing.assert_array_equal(mask, np.ones_like(img))
