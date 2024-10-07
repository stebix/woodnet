from pathlib import Path

import pytest

from woodnet.inference.utils import (parse_checkpoint,
                                     RegisteredCheckpointFilepath,
                                     EpochIntervalCheckpointFilepath)


class Test_parse_checkpoint:

    def test_with_basic_valid_qualified_fname(self):
        prefix = 'chkpt'
        qualifier = 'optimal'
        UUID = '1701-phaser-ferengi-wolf-359'
        suffix = '.pth'
        fname = '_'.join((prefix, qualifier, UUID)) + suffix
        fpath = Path('/base/dir/to/file') / fname

        parts = parse_checkpoint(fpath)
        assert isinstance(parts, RegisteredCheckpointFilepath)
        assert parts.prefix == prefix
        assert parts.qualifier == qualifier
        assert parts.UUID == UUID
        assert parts.path == fpath


    def test_with_basic_valid_unqualified_fname(self):
        prefix = 'chkpt'
        qualifier = None
        UUID = '1701-phaser-ferengi-wolf-359'
        suffix = '.pth'
        fname = '_'.join((prefix, UUID)) + suffix
        fpath = Path('/base/dir/to/file') / fname

        parts = parse_checkpoint(fpath)
        assert isinstance(parts, RegisteredCheckpointFilepath)
        assert parts.prefix == prefix
        assert parts.qualifier == qualifier
        assert parts.UUID == UUID
        assert parts.path == fpath


    def test_interval_checkpoint_with_valid_fname(self):
        prefix = 'mdl'
        midfix = 'epoch'
        epoch = 1701
        suffix = '.pth'
        fname = '-'.join((prefix, midfix, str(epoch))) + suffix
        fpath = Path('/base/dir/to/file') / fname

        parts = parse_checkpoint(fpath)
        assert isinstance(parts, EpochIntervalCheckpointFilepath)
        assert parts.prefix == prefix
        assert parts.midfix == midfix
        assert parts.epoch == epoch
        assert parts.path == fpath


    @pytest.mark.parametrize('suffix', ('', '.pth', '.yaml'))
    def test_failure_on_nonconforming_fname(self, suffix):
        fname = f'what-in_the_world-is-this_file{suffix}'
        fpath = Path('/base/dir/to/file') / fname

        with pytest.raises(ValueError):
            _ = parse_checkpoint(fpath)
