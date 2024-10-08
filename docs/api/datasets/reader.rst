Reader
======

The reader concept provides a centralized interface to retrieve the necessary *raw* data
for a dataset (raw numerical array data, i.e. the scanned image or volume)
together with its corresponding *meta* data (e.g. the class, scanner settings, etc.).

Currently, the readers for Zarr arrays and HDF5 files are implemented. If you want to
support your special use case (e.g. directories, pickled files, etc.), you can
either raise a feature request or implement an appropriate reader class here.

.. automodule:: woodnet.datasets.reader
    :members:	