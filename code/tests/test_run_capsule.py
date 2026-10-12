"""Tests for the capsule entry point"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import tifffile as tif

# log_schema is only available inside the capsule image
sys.modules.setdefault("log_schema", MagicMock())

import run_capsule  # noqa: E402
from aind_smartspim_flatfield_estimation import estimation  # noqa: E402

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def make_sidecar(suspicious=False, reasons=None, search_reason=None, chose_baseline=False):
    """Builds a sidecar like the one the package writes"""
    return {
        "suspicious": suspicious,
        "guard": {"ok": not suspicious, "reasons": reasons or []},
        "search": {"reason": search_reason, "chose_baseline": chose_baseline},
        "params": {"smoothness_flatfield": 2.0},
        "basic_config": {"fitting_mode": "ladmap"},
        "flatfield_stats": {"std": 0.1},
        "pyramid_level": 2,
    }


class FakePackage:
    """
    Replaces estimation.run_estimate. It writes the files that the
    package writes, with a sidecar chosen per channel.
    """

    def __init__(self, sidecars=None, fail_on=None):
        self.sidecars = sidecars or {}
        self.fail_on = fail_on
        self.calls = []

    def __call__(self, argv):
        output_name = argv[argv.index("--output-name") + 1]
        output_folder = Path(argv[argv.index("--output-folder") + 1])
        self.calls.append(argv)

        if output_name == self.fail_on:
            raise RuntimeError("solver failed")

        output_folder.mkdir(parents=True, exist_ok=True)
        np.save(
            output_folder.joinpath(f"flatfield_{output_name}_level0.npy"),
            np.full((6, 4), 1.0, dtype=np.float32),
        )
        sidecar = self.sidecars.get(output_name, make_sidecar())
        output_folder.joinpath(f"flatfield_{output_name}.json").write_text(json.dumps(sidecar))


class TestRunEstimation(unittest.TestCase):
    """Tests for the channel loop"""

    def setUp(self):
        """Creates the capsule data and results folders"""
        self.root = Path(tempfile.mkdtemp(prefix="unittest_"))
        self.data = self.root.joinpath("data")
        self.results = self.root.joinpath("results")
        self.data.mkdir()
        self.results.mkdir()

        shutil.copy(RESOURCES_DIR / "metadata.json", self.data / "metadata.json")
        self.data.joinpath("data_description.json").write_text(json.dumps({"name": "dataset"}))
        self.data.joinpath("derivatives").mkdir()
        tif.imwrite(self.data / "derivatives" / "DarkMaster_cropped.tif", np.zeros((4, 4)))

    def tearDown(self):
        """Removes the temporary folders"""
        shutil.rmtree(self.root, ignore_errors=True)

    def add_channel(self, channel):
        """Adds the configuration file of a channel"""
        self.data.joinpath(f"preprocess_{channel}.json").write_text(
            json.dumps({"input_data": "s3://bucket/dataset", "channel": channel})
        )

    def run_with(self, fake):
        """Runs the capsule logic with the package replaced"""
        with patch.object(estimation, "run_estimate", fake):
            run_capsule.run_estimation(self.data, self.results)

    def data_processes(self):
        """Reads the data processes written in processing.json"""
        processing = json.loads(self.results.joinpath("metadata", "processing.json").read_text())
        return processing["data_processes"]

    def assert_outputs_written(self, channel):
        """Checks the files that the destripe capsule needs"""
        for side in ("0", "1"):
            path = self.results.joinpath(f"estimated_flat_laser_{channel}_side_{side}.tif")
            self.assertTrue(path.exists(), path)
        self.assertTrue(self.results.joinpath("laser_tiles.json").exists())
        self.assertTrue(self.results.joinpath("metadata", "processing.json").exists())

    def test_missing_darkfield_fails_before_any_fit(self):
        """Tests that a missing DarkMaster stops the run early"""
        self.add_channel("ch_a")
        self.data.joinpath("derivatives", "DarkMaster_cropped.tif").unlink()
        fake = FakePackage()

        with self.assertRaises(ValueError) as context:
            self.run_with(fake)

        self.assertIn("DarkMaster_cropped.tif", str(context.exception))
        self.assertEqual(fake.calls, [])

    def test_no_channel_configuration(self):
        """Tests that a dataset without channels raises an error"""
        with self.assertRaises(FileNotFoundError):
            self.run_with(FakePackage())

    def test_clean_run_for_one_channel(self):
        """Tests a run where everything passes"""
        self.add_channel("ch_a")

        self.run_with(FakePackage())

        self.assert_outputs_written("ch_a")
        self.assertEqual(len(self.data_processes()), 1)

    def test_two_channels_call_the_package_twice(self):
        """Tests the package gets one folder per channel"""
        self.add_channel("ch_a")
        self.add_channel("ch_b")
        fake = FakePackage()

        self.run_with(fake)

        folders = sorted(argv[argv.index("--output-folder") + 1] for argv in fake.calls)
        self.assertEqual(
            folders,
            [
                str(self.results / "flatfield_estimation" / "ch_a"),
                str(self.results / "flatfield_estimation" / "ch_b"),
            ],
        )
        for argv in fake.calls:
            self.assertIn("--validate", argv)
            self.assertNotIn("--skip-search", argv)
            self.assertEqual(
                argv[argv.index("--darkfield-image") + 1],
                str(self.data / "derivatives" / "DarkMaster_cropped.tif"),
            )
        self.assertEqual(len(self.data_processes()), 2)

    def test_error_in_the_package_is_raised(self):
        """Tests that a package failure is not hidden"""
        self.add_channel("ch_a")

        with self.assertRaises(RuntimeError) as context:
            self.run_with(FakePackage(fail_on="ch_a"))

        self.assertIn("solver failed", str(context.exception))

    def test_package_that_writes_no_sidecar(self):
        """Tests that a package run without a sidecar is an error"""
        self.add_channel("ch_a")

        with self.assertRaises(FileNotFoundError):
            with patch.object(estimation, "run_estimate", lambda argv: None):
                run_capsule.run_estimation(self.data, self.results)

    def test_suspicious_flatfield_writes_outputs_then_raises(self):
        """Tests the write-then-fail policy for a suspicious flatfield"""
        self.add_channel("ch_a")
        fake = FakePackage({"ch_a": make_sidecar(suspicious=True, reasons=["std too low"])})

        with self.assertRaises(estimation.SuspiciousFlatfieldError) as context:
            self.run_with(fake)

        self.assertEqual(context.exception.reasons_by_channel, {"ch_a": ["std too low"]})
        self.assert_outputs_written("ch_a")
        self.assertIn("std too low", self.data_processes()[0]["notes"])

    def test_one_suspicious_channel_does_not_stop_the_others(self):
        """Tests that every channel runs before the error is raised"""
        self.add_channel("ch_a")
        self.add_channel("ch_b")
        fake = FakePackage({"ch_a": make_sidecar(suspicious=True, reasons=["span too low"])})

        with self.assertRaises(estimation.SuspiciousFlatfieldError) as context:
            self.run_with(fake)

        self.assertEqual(list(context.exception.reasons_by_channel), ["ch_a"])
        self.assertEqual(len(fake.calls), 2)
        self.assert_outputs_written("ch_a")
        self.assert_outputs_written("ch_b")
        self.assertEqual(len(self.data_processes()), 2)

    def test_skipped_search_writes_outputs_then_raises(self):
        """Tests the write-then-fail policy for a skipped search"""
        self.add_channel("ch_a")
        fake = FakePackage({"ch_a": make_sidecar(search_reason="over_budget")})

        with self.assertRaises(estimation.SearchSkippedError) as context:
            self.run_with(fake)

        self.assertEqual(context.exception.reason_by_channel, {"ch_a": "over_budget"})
        self.assert_outputs_written("ch_a")
        self.assertIn("over_budget", self.data_processes()[0]["notes"])

    def test_all_candidates_failed_raises(self):
        """Tests that a search without valid candidates is an error"""
        self.add_channel("ch_a")
        fake = FakePackage({"ch_a": make_sidecar(search_reason="all_candidates_failed")})

        with self.assertRaises(estimation.SearchSkippedError):
            self.run_with(fake)

    def test_search_that_kept_the_baseline_does_not_raise(self):
        """Tests that choosing the baseline smoothness is a valid search"""
        self.add_channel("ch_a")

        self.run_with(FakePackage({"ch_a": make_sidecar(chose_baseline=True)}))

        self.assert_outputs_written("ch_a")

    def test_suspicious_flatfield_has_priority_over_skipped_search(self):
        """Tests which error is raised when both problems happen"""
        self.add_channel("ch_a")
        fake = FakePackage(
            {
                "ch_a": make_sidecar(
                    suspicious=True, reasons=["std too low"], search_reason="over_budget"
                )
            }
        )

        with self.assertRaises(estimation.SuspiciousFlatfieldError):
            self.run_with(fake)

        note = self.data_processes()[0]["notes"]
        self.assertIn("std too low", note)
        self.assertIn("over_budget", note)


class TestGetChannel(unittest.TestCase):
    """Tests for reading the channel configuration"""

    def test_json_configuration(self):
        """Tests a preprocess json with S3 input"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder).joinpath("preprocess_Ex_488_Em_525.json")
            path.write_text(
                json.dumps({"input_data": "s3://bucket/dataset", "channel": "Ex_488_Em_525"})
            )

            name, channel_path = run_capsule.get_channel(path)

        self.assertEqual(name, "Ex_488_Em_525")
        self.assertEqual(channel_path, "s3://bucket/dataset/Ex_488_Em_525")

    def test_folder_configuration(self):
        """Tests a channel folder in the data folder"""
        name, channel_path = run_capsule.get_channel(Path("/data/Ex_488_Em_525"))

        self.assertEqual(name, "Ex_488_Em_525")
        self.assertEqual(channel_path, "/data/Ex_488_Em_525")


class TestMain(unittest.TestCase):
    """Tests for the logging of the stages"""

    def test_failure_is_logged_and_raised(self):
        """Tests that a failure logs stage_failure and is raised again"""
        with patch.object(run_capsule, "run_estimation", side_effect=RuntimeError("boom")):
            with patch.object(run_capsule.logger, "error") as mock_error:
                with self.assertRaises(RuntimeError):
                    run_capsule.main()

        self.assertEqual(mock_error.call_args.kwargs["extra"]["event_type"], "stage_failure")

    def test_success_is_logged(self):
        """Tests that a clean run logs stage_complete"""
        with patch.object(run_capsule, "run_estimation"):
            with patch.object(run_capsule.logger, "info") as mock_info:
                run_capsule.main()

        events = [
            call.kwargs.get("extra", {}).get("event_type") for call in mock_info.call_args_list
        ]
        self.assertIn("stage_start", events)
        self.assertIn("stage_complete", events)


if __name__ == "__main__":
    unittest.main()
