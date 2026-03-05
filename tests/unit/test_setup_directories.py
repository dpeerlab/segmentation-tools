from pathlib import Path

from tests.conftest import import_process
from segmentation_tools.utils.config import RESULTS_DIR_NAME, CHECKPOINT_DIR_NAME

_mod = import_process("000_setup_directories")
main = _mod.main


class TestSetupDirectories:
    def test_creates_directory_structure(self, tmp_dir):
        output_dir, results_dir, checkpoint_dir = main(
            output_dir_root=tmp_dir, job_title="test_job"
        )
        assert output_dir.exists()
        assert results_dir.exists()
        assert checkpoint_dir.exists()
        assert output_dir == tmp_dir / "test_job"
        assert results_dir == output_dir / RESULTS_DIR_NAME
        assert checkpoint_dir == output_dir / CHECKPOINT_DIR_NAME

    def test_recreates_existing_directory(self, tmp_dir):
        job_dir = tmp_dir / "test_job"
        job_dir.mkdir(parents=True)
        marker = job_dir / "old_file.txt"
        marker.write_text("should be deleted")

        main(output_dir_root=tmp_dir, job_title="test_job")
        assert not marker.exists()

    def test_nested_output_root(self, tmp_dir):
        nested = tmp_dir / "a" / "b" / "c"
        output_dir, results_dir, checkpoint_dir = main(
            output_dir_root=nested, job_title="deep"
        )
        assert output_dir.exists()
        assert results_dir.exists()
        assert checkpoint_dir.exists()
