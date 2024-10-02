# woodnet readme

## Table of Content

How to insert custom data into the pipeline: [Data Loading](#data-loading)
How to configure a training run: [Training Run Configuration](#training-run-configuration)

## Data Loading

The central place to inject data into the `woodnet` system is via the `dataconf.yaml` configuration file.
There you exhaustively specify all data instances as a mapping from an unique identification string
(ID) to certain metadata.
In the following, this metadata is called the dataset instance *fingerprint*.

The human-readable YAML file `dataconf.yaml` is the central tool to tell the `woodnet` framework
from where the data should be loaded.

It consists of three necessary building blocks. The `class_to_label_mapping` section, where we specify
the mapping from human-readable, semantic class names to integer numbers:
```yaml
class_to_label_mapping:
  acer : 0
  pinus : 1
```
The second building block is the instance mapping part where we can specify the dataset instances
as unique string IDs to use these IDs in other places (e.g. configurations and dataset builders).
The framework needs further information about the dataset instances, thus we need to specify
more information for every ID. This leads to the following layout:
```yaml
instance_mapping :
  awesome-unicorn:
    location: '/my/fancy/location/awesome-unicorn.zarr'
    classname: hardwood
    group: pristine
```
In the above example, we specified the dataset instance with the unique ID `awesome-unicorn`.
The fundamental data is expected to be at `'/my/fancy/location/awesome-unicorn.zarr'`.
Note that any unique string ID can be chosen here, even much more mundane like e.g. `scan-1`
for the first scan of a hypothetical series of scans.
Here, we also make first contact with the data format, namely a [`zarr`](https://zarr.dev/) array.
Later, we will take a closer look on the expected data layout, alternatives to `zarr` arrays and
ways in which we can implement additional data storage interfaces.
Going back to our `awesome-unicorn` instance, we indicated via the `classname: hardwood` attribute
that the data belongs to the `hardwood` class. We usually choose and set up classes specific for our
classification task.
The last attribute of the fingerprint is the `group` attribute. Here we have the option to specify furhter information about sub-groups in our data. Subsets of single classes may belong to a subgroup,
if some data parameters may be shared.
An illustrative example could be: We want to perform binary classification between hardwood and softwood
specied and for both classes we have a large number of samples. 


## Training Run Configuration

In this section we take a look at how to use the provided command line interface (CLI) and
configuration files (YAML) to perform a training run.

