import random
from pathlib import Path

from woodnet.loader import SliceLoader, SubvolumeLoader

instance_path = Path(
    '/home/jannik/storage/wood/chunked/CT10_Ahorn_40kV_200muA_5s_1mitt/'
)

subvolume_path = Path(
    '/home/jannik/storage/wood/chunked/CT10_Ahorn_40kV_200muA_5s_1mitt/subvol_0'
)


def test_subvolume_loader():
    loader = SubvolumeLoader()
    sv = loader.from_top_directory(instance_path, index=0)
    # print(sv)
    print(type(sv.data))
    print(sv.data.shape)

    print(sv.fingerprint)


def test_slice_loader():
    loader = SliceLoader()
    slices = loader.from_directory(subvolume_path)

    for slc in random.choices(slices, k=5):
        print(slc)