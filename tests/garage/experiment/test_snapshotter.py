from os import path as osp
import pickle
import tempfile

import pytest

from garage.experiment import Snapshotter

configurations = [
    ("all", {"itr_1.pkl": 0, "itr_2.pkl": 1}),
    ("last", {"params.pkl": 1}),
    ("gap", {"itr_2.pkl": 1}),
    ("gap_and_last", {"itr_2.pkl": 1, "params.pkl": 1}),
    ("none", {}),
]


class TestSnapshotter:
    def setup_method(self):
        # pylint: disable=consider-using-with
        self.temp_dir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        self.temp_dir.cleanup()

    @pytest.mark.parametrize("mode, files", [*configurations])
    def test_snapshotter(self, mode, files):
        snapshotter = Snapshotter(self.temp_dir.name, mode, 1)

        assert snapshotter.snapshot_dir == self.temp_dir.name
        assert snapshotter.snapshot_mode == mode
        assert snapshotter.snapshot_gap == 1

        snapshot_data = [{"testparam": 1}, {"testparam": 4}]
        snapshotter.save_itr_params(1, snapshot_data[0])
        snapshotter.save_itr_params(2, snapshot_data[1])

        for f, num in files.items():
            filename = osp.join(self.temp_dir.name, f)
            assert osp.exists(filename)
            with open(filename, "rb") as pkl_file:
                data = pickle.load(pkl_file)
                assert data == snapshot_data[num]

    def test_gap_overwrite(self):
        snapshotter = Snapshotter(self.temp_dir.name, "gap_overwrite", 2)
        assert snapshotter.snapshot_dir == self.temp_dir.name
        assert snapshotter.snapshot_mode == "gap_overwrite"
        assert snapshotter.snapshot_gap == 2

        snapshot_data = [{"testparam": 1}, {"testparam": 4}]
        snapshotter.save_itr_params(1, snapshot_data[0])
        snapshotter.save_itr_params(2, snapshot_data[1])

        filename = osp.join(self.temp_dir.name, "params.pkl")
        assert osp.exists(filename)
        with open(filename, "rb") as pkl_file:
            data = pickle.load(pkl_file)
            assert data == snapshot_data[1]

    def test_invalid_snapshot_mode(self):
        with pytest.raises(ValueError):
            snapshotter = Snapshotter(snapshot_dir=self.temp_dir.name, snapshot_mode="invalid")
            snapshotter.save_itr_params(2, {"testparam": "invalid"})

    def test_conflicting_params(self):
        with pytest.raises(ValueError):
            Snapshotter(snapshot_dir=self.temp_dir.name, snapshot_mode="last", snapshot_gap=2)

        with pytest.raises(ValueError):
            Snapshotter(snapshot_dir=self.temp_dir.name, snapshot_mode="gap_overwrite", snapshot_gap=1)

    def test_sorts_correctly(self):
        snapshotter = Snapshotter(self.temp_dir.name, "all", 2)
        snapshotter.save_itr_params(80, {"test_itr": 80})
        snapshotter.save_itr_params(120, {"test_itr": 120})
        last = snapshotter.load(self.temp_dir.name)
        assert last["test_itr"] == 120
