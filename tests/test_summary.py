from pathlib import Path

import pytest
import rich

from woodnet.datasets.summary.chex import softcheck_roi_usage, recompute_multiple_statistics

class Test_softcheck_roi_usage:

    @pytest.fixture
    def test_data(self):
        return Path('/home/jannik/storage/wood/phase/joint4/CT20.zarr')
    
    def test_softcheck_roi_usage(self, test_data):
        r = softcheck_roi_usage(test_data)

        rich.print(r)



class Test_recompute_multiple_statistics:
    @pytest.fixture
    def test_data(self):
        return Path('/home/jannik/storage/wood/phase/joint4/CT20.zarr')
    
    def test_recompute_multiple_statistics(self, test_data):
        recompute_multiple_statistics(test_data, dry_run=True)