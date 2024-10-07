"""
general utility functions for inference purposes.

@jsteb 2024
"""
import dataclasses
from pathlib import Path

@dataclasses.dataclass
class RegisteredCheckpointFilepath:
    """Parsed filepath for registered checkpoint."""
    prefix: str
    qualifier: str | None
    UUID: str
    path: Path

    def make_ID(self) -> str:
        """Produce checkpoint-specific string ID from the parts."""
        qualifier = f'{self.qualifier}_' if self.qualifier else ''
        return f'{qualifier}{self.UUID}'


@dataclasses.dataclass
class EpochIntervalCheckpointFilepath:
    """Parsed filepath for epoch interval checkpoint."""
    prefix: str
    midfix: str
    epoch: int
    path: Path

    def make_ID(self) -> str:
        """Produce checkpoint-specific string ID from the parts."""
        return f'interval_{self.epoch}'


CheckpointFilepath = RegisteredCheckpointFilepath | EpochIntervalCheckpointFilepath


CHKPT_SUFFIX: str = '.pth'


def parse_registered_checkpoint(fpath: Path) -> RegisteredCheckpointFilepath:
    """
    Attempt to parse filename as registered checkpoint.
    First tries to split into triplet, then into doublet and raises
    ValueError if both attempts fail.
    """
    fname = fpath.name.removesuffix(CHKPT_SUFFIX)
    try:
        prefix, qualifier, UUID = fname.split('_')
        return RegisteredCheckpointFilepath(prefix=prefix, qualifier=qualifier,
                                            UUID=UUID, path=fpath)
    except ValueError:
        qualifier = None
    try:
        prefix, UUID = fname.split('_')
    except ValueError as e:
        raise ValueError(f'Could not parse \'{fname}\' into prefix, qualifier and '
                         f'UUID triplet or prefix and UUID doublet.') from e
    return RegisteredCheckpointFilepath(prefix=prefix, qualifier=qualifier,
                                        UUID=UUID, path=fpath)


def parse_epoch_checkpoint(fpath: Path) -> EpochIntervalCheckpointFilepath:
    """
    Attempt to parase filename as epoch checkpoint.
    """
    fname = fpath.name.removesuffix(CHKPT_SUFFIX)
    try:
        prefix, midfix, epoch = fname.split('-')
        epoch = int(epoch)
    except ValueError as e:
        raise ValueError(f'Could not parse \'{fname}\' as epoch interval checkpoint '
                         f'file name.') from e
    return EpochIntervalCheckpointFilepath(prefix=prefix, midfix=midfix,
                                           epoch=epoch, path=fpath)


def parse_checkpoint(fpath: Path) -> CheckpointFilepath:
    try:
        return parse_registered_checkpoint(fpath)
    except ValueError as e:
        error_a = e
    try:
        return parse_epoch_checkpoint(fpath)
    except ValueError as e:
        error_b = e
    raise ValueError(f'could not parse checkpoint with filename \'{fpath.name}\'. '
                     f'Registered checkpoint parsing failed due to {error_a}. '
                     f'Epoch checkpoint parsing failed due to {error_b}')
