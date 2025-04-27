"""
Summarizing sanity checking for Zarr-based datasets.
"""
from pathlib import Path
import zarr
import rich
import tqdm.auto as tqdm

from woodnet.datasets.fingerprints import StatParams
from woodnet.datasets.utils import compute_statistics, generate_cylindrical_roi

def softcheck_roi_usage(
    fpath: Path
) -> dict[str, bool]:
    zarrobj = zarr.open(fpath, mode='r')

    results = {}

    def _visitor(name: str, obj: zarr.Group | zarr.Array):
        if isinstance(obj, zarr.Group):
            return
        
        params = StatParams.from_zarr_array(obj)

        if params.total_voxelcount is None or params.roi_voxelcount is None:
            results[name] = False
        elif params.roi_voxelcount == params.total_voxelcount:
            results[name] = False
        elif params.roi_voxelcount < params.total_voxelcount:
            results[name] = True
        else:
            raise ValueError(
                f'ROI voxel count {params.roi_voxelcount} is greater than '
                f'total voxel count {params.total_voxelcount} for {name}'
            )
        
    zarrobj.visititems(_visitor)
    return results


def recompute_statistics(
    fpath: Path,
    internal_path: str,
    dry_run: bool = True,
) -> None:
    """
    Recompute statistics for a given Zarr array.
    """
    mode = 'r' if dry_run else 'a'
    zarrobj = zarr.open(fpath, mode=mode)
    array = zarrobj[internal_path]
    if not isinstance(array, zarr.Array):
        raise TypeError(
            f'Expected zarr.Array at internal path \'{internal_path}\''
            f', got {type(array)}'
        )
    metadata = dict(array.attrs)
    data = array[...]

    roi = metadata.get('roi', None)
    if roi == 'cylindrical-center':
        mask = generate_cylindrical_roi(data.shape)
    else:
        mask = None

    stats = compute_statistics(data, mask=mask)

    if not dry_run:
        array.attrs['statistics'] = stats
    else:
        previous_statistics = metadata.get('statistics')
        rich.print(f'Would update {internal_path} with statistics')
        rich.print(f'from old statistics: {previous_statistics}')
        rich.print(f'ro new statistics: {stats}')


def recompute_multiple_statistics(
    fpath: Path,
    dry_run: bool = True,
) -> None:
    """
    Recompute statistics for all Zarr arrays in a given zarr store
    that appear to have not used the ROI during statistics computation.
    """
    softcheck_result = softcheck_roi_usage(fpath)
    for internal_path, roi_used in softcheck_result.items():
        if roi_used:
            rich.print(f'Skipping {internal_path} as it already used ROI')
            continue
        recompute_statistics(fpath, internal_path, dry_run=dry_run)


def recompute_statistics_for_directory(
    directory: Path,
    dry_run: bool = True,
) -> None:
    """
    Recompute statistics for all Zarr arrays in a given directory
    that appear to have not used the ROI during statistics computation.
    """
    wrapped_directory = tqdm.tqdm(list(directory.iterdir()), desc='objects', unit='stores')
    for item in wrapped_directory:
        if item.is_dir() and item.name.endswith('.zarr'):
            recompute_multiple_statistics(item, dry_run=dry_run)
        else:
            rich.print(f'Skipping {item} as it is not a Zarr directory')


