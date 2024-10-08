Reader
======

The reader concept provides a centralized interface to retrieve the necessary *raw* data
for a dataset (raw numerical array data, i.e. the scanned image or volume)
together with its corresponding *meta* data (e.g. the class, scanner settings, etc.).

Currently, the readers for Zarr arrays and HDF5 files are implemented. If you want to
support your special use case (e.g. directories, pickled files, etc.), you can
either raise a feature request or implement an appropriate reader class here.

The reader class is expected to conform to the following interface:

.. autoclass:: woodnet.datasets.reader.Reader
    :members:


We provide the premade readers for {``zarr``, ``hdf5``} raw data formats.
Both require a path to the file or directory containing the raw data and an
internal path specification to the data within the file.
This allows the bundling of multiple versions of the same dataset in a single file.

.. autoclass:: woodnet.datasets.reader.HDF5Reader
    :members:

.. autoclass:: woodnet.datasets.reader.ZarrReader
    :members:

.. autofunction:: woodnet.datasets.reader.deduce_reader_class

.. autofunction:: woodnet.datasets.reader.read_data_from_hdf5

.. autofunction:: woodnet.datasets.reader.read_data_from_zarr

.. autofunction:: woodnet.datasets.reader.read_fingerprint_from_hdf5

.. autofunction:: woodnet.datasets.reader.read_fingerprint_from_zarr

