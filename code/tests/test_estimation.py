"""Tests for the layer between the capsule and aind-flatfield-correction"""

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import tifffile as tif
from aind_smartspim_flatfield_estimation import estimation


def make_sidecar(suspicious=False, reasons=None, search_reason=None):
    """Builds a sidecar like the one the package writes"""
    return {
        "suspicious": suspicious,
        "guard": {"ok": not suspicious, "reasons": reasons or []},
        "search": {"reason": search_reason, "chose_baseline": False},
        "params": {"smoothness_flatfield": 2.0},
    }


class TestBuildEstimateArgv(unittest.TestCase):
    """Tests for the package arguments"""

    def build(self, **kwargs):
        """Builds the arguments with fixed values"""
        return estimation.build_estimate_argv(
            channel_path="s3://bucket/dataset/Ex_488_Em_525",
            darkfield_path="/data/derivatives/DarkMaster_cropped.tif",
            output_folder="/results/flatfield_estimation/Ex_488_Em_525",
            output_name="Ex_488_Em_525",
            **{"pyramid_level": 2, "max_search_minutes": 480, **kwargs},
        )

    def value_of(self, argv, flag):
        """Gets the value that follows a flag"""
        return argv[argv.index(flag) + 1]

    def test_validate_is_on_by_default(self):
        """Tests the validation figures are written by default"""
        self.assertIn("--validate", self.build())

    def test_validate_can_be_turned_off(self):
        """Tests that validate=False removes the flag"""
        self.assertNotIn("--validate", self.build(validate=False))

    def test_smartspim_settings(self):
        """Tests the settings that belong to SmartSPIM"""
        argv = self.build()

        self.assertEqual(argv[0], "s3://bucket/dataset/Ex_488_Em_525")
        self.assertEqual(self.value_of(argv, "--pyramid-level"), "2")
        self.assertEqual(self.value_of(argv, "--tile-pattern"), r"\.zarr$")
        self.assertEqual(
            self.value_of(argv, "--darkfield-image"),
            "/data/derivatives/DarkMaster_cropped.tif",
        )
        self.assertEqual(self.value_of(argv, "--output-name"), "Ex_488_Em_525")
        self.assertEqual(self.value_of(argv, "--method"), "fit")

    def test_pyramid_level_and_budget_come_from_the_caller(self):
        """Tests that the settings are arguments, not constants of the module"""
        argv = self.build(pyramid_level=3, max_search_minutes=120)

        self.assertEqual(self.value_of(argv, "--pyramid-level"), "3")
        self.assertEqual(self.value_of(argv, "--max-search-minutes"), "120")
        self.assertFalse(hasattr(estimation, "PYRAMID_LEVEL"))
        self.assertFalse(hasattr(estimation, "SEARCH_BUDGET_MINUTES"))

    def test_search_cannot_be_skipped(self):
        """Tests that the arguments never skip the parameter search"""
        self.assertNotIn("--skip-search", self.build())
        self.assertNotIn("--skip-search", self.build(validate=False))
        self.assertNotIn("--skip-search", self.build(pyramid_level=3))


class TestRunEstimate(unittest.TestCase):
    """Tests for the call to the package"""

    def test_calls_package_main_with_arguments(self):
        """Tests that the arguments go to the package unchanged"""
        fake_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {"aind_flatfield_correction.core.basicpy.estimate": fake_module},
        ):
            estimation.run_estimate(["a", "b"])

        fake_module.main.assert_called_once_with(["a", "b"])


class TestRunEstimateExit(unittest.TestCase):
    """Tests for SystemExit raised by the package"""

    def run_with_exit(self, code):
        """Runs run_estimate with a package that exits with the code"""
        fake_module = MagicMock()
        fake_module.main.side_effect = SystemExit(code)
        with patch.dict(
            "sys.modules",
            {"aind_flatfield_correction.core.basicpy.estimate": fake_module},
        ):
            estimation.run_estimate(["a"])

    def test_bad_arguments_become_an_exception(self):
        """Tests that argparse exits are catchable as Exception"""
        with self.assertRaises(estimation.FlatfieldEstimationError):
            self.run_with_exit(2)

    def test_clean_exit_is_not_an_error(self):
        """Tests that exit code 0 or None is accepted"""
        self.run_with_exit(0)
        self.run_with_exit(None)


class TestReadEstimationReport(unittest.TestCase):
    """Tests for reading the package sidecar"""

    def setUp(self):
        """Creates a temporary folder"""
        self.folder = Path(tempfile.mkdtemp(prefix="unittest_"))

    def tearDown(self):
        """Removes the temporary folder"""
        shutil.rmtree(self.folder, ignore_errors=True)

    def write_sidecar(self, sidecar):
        """Writes a sidecar and returns its path"""
        path = self.folder.joinpath("flatfield_ch.json")
        path.write_text(json.dumps(sidecar))
        return path

    def test_clean_sidecar(self):
        """Tests a flatfield that passed and a search that ran"""
        report = estimation.read_estimation_report(self.write_sidecar(make_sidecar()))

        self.assertFalse(report.suspicious)
        self.assertEqual(report.guard_reasons, [])
        self.assertIsNone(report.search_reason)
        self.assertFalse(report.search_skipped)
        self.assertEqual(report.params, {"smoothness_flatfield": 2.0})

    def test_suspicious_sidecar(self):
        """Tests that the guard reasons are returned"""
        sidecar = make_sidecar(suspicious=True, reasons=["std 0.01 < floor 0.03"])
        report = estimation.read_estimation_report(self.write_sidecar(sidecar))

        self.assertTrue(report.suspicious)
        self.assertEqual(report.guard_reasons, ["std 0.01 < floor 0.03"])

    def test_skipped_search(self):
        """Tests that a skipped search is reported"""
        sidecar = make_sidecar(search_reason="over_budget")
        report = estimation.read_estimation_report(self.write_sidecar(sidecar))

        self.assertTrue(report.search_skipped)
        self.assertEqual(report.search_reason, "over_budget")

    def test_search_that_kept_the_baseline_is_not_skipped(self):
        """Tests that choosing the baseline smoothness is a valid search"""
        sidecar = make_sidecar()
        sidecar["search"]["chose_baseline"] = True
        report = estimation.read_estimation_report(self.write_sidecar(sidecar))

        self.assertFalse(report.search_skipped)

    def test_suspicious_and_skipped_search(self):
        """Tests that both problems are reported together"""
        sidecar = make_sidecar(
            suspicious=True, reasons=["span too low"], search_reason="all_candidates_failed"
        )
        report = estimation.read_estimation_report(self.write_sidecar(sidecar))

        self.assertTrue(report.suspicious)
        self.assertTrue(report.search_skipped)
        self.assertEqual(report.search_reason, "all_candidates_failed")

    def test_sidecar_without_search_section(self):
        """Tests a sidecar where the search value is null"""
        sidecar = make_sidecar()
        sidecar["search"] = None
        report = estimation.read_estimation_report(self.write_sidecar(sidecar))

        self.assertFalse(report.search_skipped)

    def test_missing_sidecar(self):
        """Tests that a missing sidecar raises an error"""
        with self.assertRaises(FileNotFoundError):
            estimation.read_estimation_report(self.folder.joinpath("missing.json"))


class TestWriteSideFlats(unittest.TestCase):
    """Tests for the flatfield tif files"""

    def setUp(self):
        """Creates a temporary folder and a flatfield"""
        self.folder = Path(tempfile.mkdtemp(prefix="unittest_"))
        self.flatfield = np.random.rand(8, 6).astype(np.float64) + 0.5
        self.flatfield_path = self.folder.joinpath("flatfield_ch_level0.npy")
        np.save(self.flatfield_path, self.flatfield)

    def tearDown(self):
        """Removes the temporary folder"""
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_file_name_has_channel_and_side(self):
        """Tests the file name of one side"""
        path = estimation.write_flat(self.flatfield_path, "Ex_488_Em_525", "1", self.folder)

        self.assertEqual(Path(path).name, "estimated_flat_laser_Ex_488_Em_525_side_1.tif")
        self.assertTrue(Path(path).exists())

    def test_file_is_float32_with_the_same_values(self):
        """Tests the dtype, shape and values of the written flatfield"""
        written = tif.imread(estimation.write_flat(self.flatfield_path, "ch", "0", self.folder))

        self.assertEqual(written.dtype, np.float32)
        self.assertEqual(written.shape, self.flatfield.shape)
        np.testing.assert_allclose(written, self.flatfield, rtol=1e-6)


class TestBuildTilePattern(unittest.TestCase):
    """Tests for the tile selection regex"""

    def test_matches_only_the_given_tiles(self):
        """Tests the pattern against real tile folder names"""
        pattern = re.compile(estimation.build_tile_pattern(["345020_229180", "409820_177340"]))

        self.assertTrue(pattern.search("345020_229180.ome.zarr"))
        self.assertTrue(pattern.search("409820_177340.ome.zarr"))
        self.assertFalse(pattern.search("345020_151420.ome.zarr"))
        # No partial matches of an id inside a longer name
        self.assertFalse(pattern.search("1345020_229180.ome.zarr"))
        self.assertFalse(pattern.search("345020_2291800.ome.zarr"))

    def test_no_tiles_raises(self):
        """Tests that an empty side cannot build a pattern"""
        with self.assertRaises(ValueError):
            estimation.build_tile_pattern([])


class TestErrors(unittest.TestCase):
    """Tests for the error messages"""

    def test_search_skipped_error_names_channel_and_reason(self):
        """Tests the error message has the channel and the reason"""
        error = estimation.SearchSkippedError({"ch_a": "over_budget"})

        self.assertIn("ch_a", str(error))
        self.assertIn("over_budget", str(error))
        self.assertIsInstance(error, estimation.FlatfieldEstimationError)


if __name__ == "__main__":
    unittest.main()


class TestStrictSidecar(unittest.TestCase):
    """Tests for the validation of the sidecar"""

    def setUp(self):
        """Creates a temporary folder"""
        self.folder = Path(tempfile.mkdtemp(prefix="unittest_"))
        self.path = self.folder.joinpath("flatfield_ch.json")

    def tearDown(self):
        """Removes the temporary folder"""
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_renamed_suspicious_key_raises(self):
        """Tests that a missing key is not read as not suspicious"""
        sidecar = make_sidecar()
        sidecar["flagged"] = sidecar.pop("suspicious")
        self.path.write_text(json.dumps(sidecar))

        with self.assertRaises(ValueError) as context:
            estimation.read_estimation_report(self.path)

        self.assertIn("suspicious", str(context.exception))

    def test_null_guard_is_accepted(self):
        """Tests a sidecar with guard set to null"""
        sidecar = make_sidecar()
        sidecar["guard"] = None
        self.path.write_text(json.dumps(sidecar))

        report = estimation.read_estimation_report(self.path)

        self.assertEqual(report.guard_reasons, [])
