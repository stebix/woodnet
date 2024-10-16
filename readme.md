# woodnet

[![Documentation Status](https://readthedocs.org/projects/woodnet/badge/?version=latest)](https://woodnet.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# Introduction

Welcome to the `woodnet` repository. This code infrastructure arose as part of the project to further establish
advanced computed tomography imaging for wood scientific applications.
The package is aimed at providing a data processing and deep learning infrastructure for imaging data in the context of wood scientific and wood anatomical classification.
The imaging modality is thereby not baked in!
In our founding project, we used it for 3D (volumetric), 2.5D (orthogonal channel-wise concatenated) and 2D (planar) imaging data stemming from advanced, sub-μm computed tomography data.
The pipeline can be used directly for photographic (e.g. planar RGB), microscopic cross-sectional (e.g. 2.5D multi-channel) and magnetic resonance imaging data (e.g. 3D volumetric).

Core features encompass facilities for data loading and transforming, data splitting for cross validation purposes, fitting deep learning models (logging, checkpointing, metrics visualization with `tensorboard`) and thorough evaluation of these models with input data transformations.

## Overview: Quick Links

- How to insert custom data into the pipeline: [Data Loading](#data-loading)
- How to configure a training run: [Training Run Configuration](#training-run-configuration)
- How to launch a training experiment: [Training Experiment Launch](#run-training-experiment)

## Getting Started

### Installation

To install the project into your local machine, first pull the repository to a location you desire.

```bash
cd /desired/location
git pull https://github.com/stebix/woodnet.git
```

We recommend to install the package using some sort of virtual environment. For heavy deep-learning machinery with compiled/C/C++/CUDA components like PyTorch,
we recommend using `conda` or the modern and faster sibling `mamba`. If you decide to use `mamba`, supplant the `conda` command with `mamba`.

As a first step, create the environment from the provided `environment.yaml` file in the repository. This will preinstall all necessary dependencies.
With the current working directory being the pulled repository, we can execute

```bash
conda env create -f environment.yaml
```

> [!NOTE]
> This installation command assumes that you have a GPU installed. If this is not
> the case you may want to switch to CPU-only Pytorch. For this, just use the CPU-wise command below. For special combinations of platforms and CUDA packages, please consult the installation instructions of the [official documentation](https://pytorch.org/)
> 
```bash
conda env create -f environment-cpuonly.yaml
```

The default name for the newly created environment is `woodanalysis`. If we want another environment name, we can just modify the first line of the environment file.
We than need to activate the environment and may inspect the correct installation of the required packages can be inspected via the `list` command.

```bash
conda activate woodanalysis
conda env list
```

Of course, modifications to the environment name have to be respected here.
Then, we can install a local editable version of the package via `pip` using the command

```bash
pip install --editable .
```

Then the package is importable as any Python package for sessions started within this environment 🎉
This installation process also allows you to use the package or parts of it as a (modifiable) library in the context different from the current use cases.

>[!NOTE]
> For future releases, we plan to switch to a full PyPI and `anaconda` package release. But currently, the above clone + install method is the recommended one!
> Any updates can be retrieved by navigating to the repository and pulling in the changes. Via the editable install, these are then directly available environment-wide.


# Usage

Here we will learn how to use the core functionality of the `woodnet` pipeline as a (command line) tool to perform training experiments and make predictions and evaluations with trained models.
After successful installation, the package provides you with three command line tools: `net`, `refolder` and `benchmark` that can be directly accessed from the environment and shell the `woodnet` package is installed to.
The basic syntax with some hints for help is as follows:

```bash
net [VERB] [ARGS]               # basic invocation
net -h/--help                   # display help for verbs
net [VERB] -h/--help            # display help for the specific selected verb
```

The available verbs are:

- ``train`` : Run a single training experiment configured via a YAML file.
- ``batchtrain`` : Run multiple training experiments sequentially (multiple YAML files)
- ``predict`` : Run a single prediction task configured via a YAML file.
- ``evaluate`` : Evaluate and aggregate over multiple training experiments in the context of cross validation.

For the core of the verbs, i.e. ``train``, ``batchtrain`` and ``predict``, simply a configuration file is required. For this, see the corresponding [sections](#run-training-experiment).
The ``evaluate`` verb requires more arguments, namely the indication of the basal training experiment directory where the different fold-wise subexperiments live.
For this, look [here](#run-evaluation-experiment) or consult the detailed CLI documentation
at [`woodnet CLI documentation`](https://woodnet.readthedocs.io/en/latest/usage.html).

If we are more interested in using parts of the code as a library, then the documentation [API reference](https://woodnet.readthedocs.io/en/latest/api/modules.html) over there might be more appropriate.
> [!WARNING]
> Please note however that the intended use for the `woodnet` framework and pipeline is still in flux and we intend to adapt substantially to own further work and community wishes. So please do not count too much on API stability (yet).

## Run Training Experiment

The necessary prerequisites for smoothly running a training experiment are a valid data loading setup (the system must know here to find our training end evaluation data) and a training configuration file
(the system must know the precise parameters for the manifold numbers of settings present for a deep learning experiment).
If we set up our training experiment configuration with all necessary [components](#components) at a location of our choice, we can run the task via the command line invocation

```bash
net train /path/to/training-configuration.yaml
```

Then the training starts and runs according to the configurations settings.
We need to keep the terminal open and can check for progress reporting via progress bars.

### Training Monitoring

When the training tasks runs successfully, we can monitor the - possibly long running - training process live.
For this, we can either read the emitted log records in the log file (with a tool like `lnav`) or use a visualization tool.
The `woodnet` package uses [`tensorboard`](https://www.tensorflow.org/tensorboard) to monitor the training process.
To use it, in a separate terminal that has the `woodanalysis` environment enabled, navigate to the defined experiment directory

```bash
cd /path/to/awesome/experiment-dir
```

from our training experiment configuration. Then we can start the tensorboard server with

```bash
tensorboard --logdir=.
```

Accessing the provided URL or [`http://localhost:6006/`](http://localhost:6006/)
opens the live dashboard.
<img src="https://pytorch.org/tutorials/_images/tensorboard_scalars.png" width=50% height=50%>
*Image from PyTorch Docs*

## Run Evaluation Experiment

To thoroughly evaluate an experiment with cross validation, we can use the CLI tooling again. The ``woodnet`` package facilitates automated evaluation (model loading & configuration fold-wise and checkpoint-wise) and evaluation with
input data transformations. Results are stored on the file system and can be aggregated with further tools.
Of course, it is necessary that we have completed $N >= 2$ successful training experiments with cross validation to evaluate them.
The evaluation pipeline expects that the individual training experiments in the cross validation context are laid out in a canonical structure.
The structure is illustrated below:

```
.
└── cv-experiment-basedir/
    ├── fold-1/
    │   ├── logs/
    │   └── checkpoints/
    ├── fold-2/
    │   ├── logs/
    │   └── checkpoints/
    ├── ...
    ├── fold-N/
    │   ├── checkpoints/
    │   └── logs/
    └── inference/                  # newly created on first evaluation task
        └── timestamp-1/            # subsequent runs get unique timestamps
            ├── evaluation.json
            ├── timestamp.log
            └── ...
```

For such a training experiment layout, we can run the full evaluation again via CLI via

```bash
net evaluate /path/to/experiment-basedir transform-template 
```

The second argument is the template name (name for builtin, path for template file anywhere on the system) specifying the transformations to use for the robustness evaluation.
Then the system will perform predictions and evaluate all models (could be many due to the sampling/saving of model states) of all folds (determined by our CV strategy) with all transformations (set in the transform template) applied to the input data and aggregate the results in an inference directory created on the level of the `fold-N` directories.
For every evaluation run, a new subdirectory with the timestamp of the run is created.
We can then process, analyze and visualize the aggregated performance metrics to gain insights over model performance and potential performance degradation for input data transformation.

# Components

In this section, we look at the different components of the model and data pipeline.
We want to provide insights about possibilities to configure the package.
The main entry point for primary usage is the [data loading section](#data-loading) where instructions about injecting your data (e.g. scanned volumes, scanned planar images or microscopy data) into the system is provided.
The following sections are concerned with explaining the configuration files
to [control training experiments](#training-run-configuration) and performing prediction and evaluation tasks.

## Data Loading

The central place to inject data into the `woodnet` system is via the `dataconf.yaml` configuration file.

You have two ways to insert the path to the datac onfiguration file. Variant A is by using the CLI application. Variant B is by modification or creation of the `.env` environment file in the repository root on your machine.

We can point the `woodnet` pipeline by inserting the location (absolute path ideally)
in a `.env` file located in the repository root.

```bash
woodnet-repository
  - .env                # create this here or modify it
  - ...
```

There you exhaustively specify all data instances as a mapping from an unique identification string
(ID) to certain metadata.
In the following, this metadata is called the dataset instance *fingerprint*.

The human-readable YAML file `dataconf.yaml` is the central tool to tell the `woodnet` framework
from where the data should be loaded.

It consists of three necessary building blocks. The `class_to_label_mapping` section, where we specify
the mapping from human-readable, semantic class names to integer numbers:

```yaml
class_to_label_mapping:
  softwood : 0
  hardwood : 1
```

The second building block is the instance mapping part where we can specify the dataset instances
as unique string IDs to use these IDs in other places (e.g. configurations and dataset builders).
Another advantage of this central registration of datasets is the possibility to automatically
split our datasets into disjoint training and validation sets and transform a training configuration file correspondingly. For further information on cross validation functionality, head over to the small tutorial [chapter](#cross-validation-tooling).
The framework needs further information about the dataset instances, thus we need to specify
more information for every ID. This leads to the following layout:

```yaml
instance_mapping :

  awesome-unicorn:
    location: '/my/fancy/location/awesome-unicorn.zarr'
    classname: hardwood
    group: pristine

  scan-ef04:
    location: '/my/fancy/location/scan-ef04.zarr'
    classname: softwood
    group: pristine

  scan-c07f:
    location: '/my/other/data/location/scan-c07f.zarr'
    classname: hardwood
    group: withered

  jean-luc-picard:                                      # unique ID can be different from file name
    location: '/another/datasource/scan-x07j.zarr'      # but similarity can be a good idea
    classname: hardwood
    group: withered
```

In the above example, we specified the first dataset instance with the unique ID `awesome-unicorn`.
Of course, arbitrarily many datasets - each with unique ID - can be specified with the file.
The fundamental data is expected to be at `'/my/fancy/location/awesome-unicorn.zarr'`.
Note that any unique string ID can be chosen here, even much more mundane like e.g. `scan-1`
for the first scan of a hypothetical series of scans.
Here, we also make first contact with the data format, namely a [`zarr`](https://zarr.dev/) array.
Later, we will take a closer look on the expected data layout, alternatives to `zarr` arrays and
ways in which we can implement additional data storage interfaces.
Going back to our `awesome-unicorn` instance, we indicated via the `classname: hardwood` attribute
that the data belongs to the `hardwood` class. We usually choose and set up classes specific for our
classification task.
The last attribute of the fingerprint is the `group` attribute. Here we have the option to specify further information about sub-groups in our data. Subsets of single classes may belong to a subgroup,
if some data parameters may be shared.
An illustrative example could be: We want to perform binary classification between hardwood and softwood
species and for both classes we have a large number of samples. For both classes, we obtained
samples from freshly logged wood that we mark with the `group: pristine` attribute.
We additionally got samples that were exposed to the elements and mark these with the
`group: withered` attribute. We can use the `group` data instance attribute during the creation of the cross-validation splits of the specified instances into the training set and the validation set.
In addition to the "default" variant of class-stratified `k`-fold cross-validation we may then
employ group-wise `k`-fold cross-validation. Then we can evaluate whether the model is able/flexible/intelligent enough to generalize across groups.

The last necessary building block for the data configuration file is the specification of
an internal path. The internal path is the last bit of information of the route to
the basic numerical array data. Both current natively supported array formats (`zarr` and `hdf5`) support internal paths, allowing the bundling of multiple array-wise dataset in a single object. Thus, we need to sepcify the internal path for our datasets like so:

```yaml
internal_path: 'group/dataset'
```

You can choose this as you like, but for the time being it should be global for the subset of arrays we are using in training and evaluation.
A full exemplary template of the data configuation file is shown

```yaml
class_to_label_mapping:
  softwood : 0
  hardwood : 1

instance_mapping:

  awesome-unicorn:
    location: '/my/fancy/location/awesome-unicorn.zarr'
    classname: hardwood
    group: pristine

  scan-ef04:
    location: '/my/fancy/location/scan-ef04.zarr'
    classname: softwood
    group: pristine

  scan-c07f:
    location: '/my/other/data/location/scan-c07f.zarr'
    classname: hardwood
    group: withered

  jean-luc-picard:                                      # unique ID can be different from file name
    location: '/another/datasource/scan-x07j.zarr'      # but similarity can be a good idea
    classname: hardwood
    group: withered


internal_path: 'group/dataset'
```

For your own experiments, just introduce your datasets into the file and use the unique identifiers in the configuration file to start a training experiment.

## Training Run Configuration

In this section we take a look at how to use the provided command line interface (CLI) and
configuration files (YAML) to perform a training run.
We dissect an exemplary training configuration file by taking a closer look at each individual
section component.

### General Block

This block sets the output directory for the training experiment and the training device. It generally looks like so:

```yaml
experiment_directory: /path/to/awesome/experiment-dir
device: cuda:1                                            
```

#### Training Directory

The training directory (i.e. `experiment-dir` in the above example) is the central collection location where all permanent artifacts of our
training experiment are saved.
The permanent artifacts are:

- Trained model weight checkpoints: this is the primary result of our experiment! A `checkpoints` subdirectory contains all checkpoints files.

- Log file: a large number of settings, events and stuff is logged for later inspection in a text log file. This file is located in a `logs` folder inside the `experiment_directory`.

- Configuration file: the configuration file for the training experiment is backed up in this directory as well. This enables the analysis of the experiment later on (very handy!). This file is also located in the `logs` directory.

- `tensorboard` log file: We use this library to visualize and analyze the training experiment on the fly. More on this later. This file is also located in the `logs` directory.

This leads to the layout shown below. Note that the file names may differ (timestamps, etc.).

```
.
└── experiment-dir/
    ├── checkpoints/
    │   ├── chkpt_$UUID.pth            # model checkpoint file
    │   ├── ...
    │   └── chkpt_$UUID-N.pth          # another model checkpoint file
    └── logs/
        ├── $timestamp.log             # log file for full training experiment
        ├── backup_training_configuration.yaml
        └── events.out.tfevents.$UUID  # tensorboard event file
```

The directory will be created if it is not present. Due to the uniqueness of all above artifacts to a single training experiments it is highly recommended to choose a new training directory for each individual training experiment.

#### Device

The device option lets us choose the device on which we want to perform the training experiment calculation. The common options are `cpu` for (often infeasibly slow) central processing unit (CPU) training or `cuda` for accelerated graphic processing unit (GPU) training. For systems that sport multiple GPUs, we can use `cuda:$N` with `$N` indicating an appropriate integer that pins the specific GPU in our system on which we desire the training experiment to run on.

### Model Block

In the model block, we configure our core deep learning model.
In general, we can set all user-facing parameters in the initializer (i.e. `__init__` method) of the model class here. Additionally, model ahead-of-time (AOT) and just-in-time (JIT) compilation
flags can be set here in the optional `compile` subconfiguration. For more information on AOT and JIT-functionality via `torch.compile` please consider the [PyTorch docs](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).
> [!CAUTION]
> The very useful compilation functionality is currently sadly not 
> supported under Windows. This is caused by the underlying `triton` compiler backend. See e.g. in the [GitHub issue](https://github.com/pytorch/pytorch/issues/122094).
> A fix/upgrade might however be released soon 🤞
Back to specifying our deep learning model.
A typical model block may look like this:

```yaml
model:
  name: ResNet3D          # model class and settings go here
  in_channels: 1
  compile:                # optional model compilation settings
    enabled: True         # using this can speed up training and prediction tasks
    dynamic: False
    fullgraph: False
```

In this example, we selected the `ResNet3D` from our model zoo and configured it to have a single input channel. Single channel data is typical for monochromatic computed tomography data. For light microscopy data, we may encounter multi channel data due to the separate measurement of red, green and blue intensities (RGB) in a photographic sensor.
We also (optionally) set the model compilation flag. In the above example, the model will be compiled at the first iteration at the cost of a small, singular latency increase and the benefit of substantial acceleration during following iterations.
>[!TIP]
> If we want to use custom model implementations, we can inject implementations into the package
> or modify files.
> So if another architecture is needed, we can head over to the section on injecting [custom models](#custom-models).
> We also plan to support more models directly in the future 🚀

### Optimizer Block

This block specifies optimizer, i.e. the algorithm with which we compute our gradients to perform the descent step each iteration. Here, you may select from all `PyTorch`-provided algorithms that live in
the [`torch.optim`](`https://pytorch.org/docs/stable/optim.html#algorithms`) namespace. Popular choices include `Adam` and `SGD`.

```yaml
optimizer:
  name: Adam
  learning_rate: 1e-3
```

The most important optimizer hyperparameter, the step size during gradient descent, is the `learning_rate`. It must always be provided.
Any further keyword arguments are passed through to the optimizer instance at initialization time.

### Loss Function Block

In this block we can select the loss function. Similar to the optimizer block, we have full access to the Pytorch-supplied [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions).

```yaml
loss:
  name: BCEWithLogitsLoss
  reduction: mean
```

Again, the loss function class is selected via the `name` field that must match the desired loss function class of Pytorch.
Any further keyword arguments are passed trough to the class initializer function.

### Trainer Block

The trainer block can be utilized to set core parameters of the training experiment run.
Major settings are explained via comments in the following exemplary trainer configuration:

```yaml
trainer:
  # select the trainer class via its string name
  name: Trainer
  # set the log frequency of core model metrics
  log_after_iters: 1000
  # set the frequency for performing a validation run
  validate_after_iters: 2500
  # set the maximum allowed number of epochs and iterations
  max_num_epochs: 500
  max_num_iters: 175000
  # select the validation metric and indicate whether a higher or lower score is better
  # for the current setting 'classification accuracy' (ACC), obviously higher is better
  validation_metric: ACC
  validation_metric_higher_is_better: True
  # configure the top-k-cache of of model weights we want to retain for this training experiment 
  score_registry:
    name: Registry
    capacity: 4
    score_preference: higher_is_better
  # advanced training experiment debugging: set parameter/gradient/... 
  # logging and visualization in tensorboard 
  parameter_logger:
    name: HistogramLogger
```

For a validation run, the training is paused and predictions for all validation data instances will be performed. The result of this run (i.e. the validation metric score) is reported to the log file and sent to the tensorboard inspection tool.
Also, the model weights are saved as a checkpoint if the score for a validation run is optimal or in the top-`k`-optimal range.
This setting influences the number of checkpoint files we may encounter in our ``checkpoints`` directory (see this [section](#run-evaluation-experiment)) after the conclusion of the run.
We can set the maximum number of iterations and epochs as an exit condition for conclusion of the training experiment. Note that the system exits the experiment run as soon as the first of both criterions is fulfilled.

### Loaders Block

The loaders block is concerned with configuring the data loading.
In the global block, we can configure general settings. The two following subblocks
are concerned with settings that are specific to the data loading and processing within the
two distinct phases, namely the `train` (training) phase and the `val` (validation) phase.  

#### Global Loaders Subblock

We can select the dataset class via the `dataset` attribute in the global loaders subblock.
This is the primary setting for selection of the 2D, 2.5D and 3D formulations of the pipeline.
The dataset classes and their accompanying builder classes implement the loading of the raw data from the file system into the main memory and their partitioning into appropriately shaped elements.
For `TileDataset`, we would receive subvolume chunks formed according to `tileshape` like $(t_z, t_y, t_x)$.
For `TriaxialDataset`, we would receive concatenated triaxial slices of the form $(3, t_y, t_x)$.
For `TiledEagerSliceDataset`, we would receive planar slices of the form $(t_y, t_x)$.

```yaml
loaders:
  # select the dataset class
  dataset: TileDataset
  # set the size of the of the subvolume or slice-tile
  tileshape: [256, 256, 256]
  # batch size setting - tune to VRAM memory availability
  batchsize: 2
  # set multiprocessing worker count for data loaders
  num_workers: 0
  # toggle memory pinning for the data loader
  pin_memory: True
```

> [!WARNING]
> Note that we have to make sure that the data dimensionality (2D, 2.5D, 3D) matches the model dimensionality.
> Otherwise we may get shape mismatch errors at the beginning of the training experiment. 

The `num_workers` setting allows us to set the worker process count for data loading. It should be a nonnegative integer and the setting `0` symbolizes single-thread data loading (everything happens in the main thread). The performance implications of this setting can be substantial (both positive and negative) and are interdependent with other aspects/settings (i.e. data processing and augmentation, rad speeds, ...). To get sensible orientation data for optimal settings, we may use the `benchmark` CLI tool provided by the `woodnet` package.
The `pin_memory` setting toggles the usage of pinned, i.e. non-paged memory for the Pytorch CPU-based tensors. Using pinned memory can increase data transfer performance in certain scenarios.

#### Training Loader Subblock

The training loader subblock must be included in the global loaders block.
Here we can set the dataset instances that are used for training the model weights by writing the desired instance IDs into the `instances_ID` list.
For training data augmentation, we can also specify one or as many as desired training data transformations as elements of a list under the key `transform_configurations`.

```yaml
train:
  # select the training data instances via the unique identifiers that were set in the
  # data configuration file
  instances_ID: [awesome-unicorn, scan-ef04, scan-c07f]

  transform_configurations:
    - name: Normalize
      mean: 1
      std:  0.5

    - name: GaussianNoise
      mean: 0.0
      std: 0.5
      p_execution: 0.33
```

For the transformations, we can again make use of the simple `keyword : value` syntax of YAML. Minimally, the name attribute of the transform is required to find the corresponding class in the code.
We can use custom transformation classes that are implemented inside the namespace/module `woodnet.transformations.transforms`. If we want to randomize the choice of transformations we can employ the container classes located in `woodnet.transformations.container`.
An additional set of diverse transformations is provided via the [MONAI](https://docs.monai.io/en/stable/transforms.html#vanilla-transforms) third party package. These transforms are also automatically recognized via the name attribute (must exactly match the class name).
The configuration is again performed via keyword passthrough.
Thus, the keyword argument dictionary `{'mean' : 1, 'std' : 0.5}` is passed tp the `Normalize`
class initializer.

#### Validation Loader Subblock

The validation loader section is in principle very similar to the training loaders subblock. An exemplary instance is given below. In practice, a more extensive/substantial validation set is of course advisable.

```yaml
val:
  instances_ID: [jean-luc-picard]

  transform_configurations:
    - name: Normalize
      mean: 1.1
      std:  0.52
```

Usually, the transformations applied to the validation data elements differ from the training data transformations.
Firstly, we have to compute features like mean and standard deviation differently for every subset to avoid premature feature engineering.
Secondly, the generation of synthetic data via augmentation is a beneficial procedure applied in the training phase. However, in the validation phase usually unaugmented data is utilized.  

## Cross Validation Tooling

Cross validation (CV) is a crucial technique for improving the reliability of our deep learning models, especially when we are working with limited data. In the small data regime, the hazard of our models to overfit or to succumb to selection bias, meaning they perform well on training data but poorly on unseen data, is relatively larger.
Instead of training on just one split of the data, we divide our dataset into multiple "folds" and train the model multiple times, each time using a different fold as the validation set.
This ensures that the model performance is assessed on a variety of data splits, reducing the risk of overfitting and over-optimistically evaluating the performance of our model.

> [!NOTE]
> The CV experiment basically reduces to performing quite similar training experiments with different unique dataset element IDs in the training and validation section, i.e. *ceteris paribus*.
> Thus, other (hyper-) parameters should be kept the same.  

The `woodnet` machinery provides some convenience tools to quickly perform cross validation for our training experiments to mitigate tedious manual editing and potential errors.
First, it offers automated splitting machinery operating automatically utilizing the dataset instances we declared in the instance mapping.
The ``refolder`` tool then allows the creation of new configuration files from templates
with training and validation dataset instances set according to the split!
This simplifies performing cross validation experiments, since we can start with a basal configuration file, produce `N` fold-wise configurations from it and run the experiments sequentially with the `batchtrain` command. For a detailed specification of both commands head over to the [CLI documentation](https://woodnet.readthedocs.io/en/latest/usage.html#net-cli).

> [!TIP]
> To produce a set of fold-wise CV configurations, make sure to change both the training-test splits
> and the experiment directory in the configuration.
> The experiment directory setup should be such that the canonical directory structure shown [here](#run-evaluation-experiment) is reproduced.

The core training-validation split of the unique IDs can be performed with on of two currently supported splitting techniques:

- Stratified `k`-fold cross-validation is a variation of `k`-fold cross-validation that ensures each fold preserves the proportion of classes in the original dataset. In standard `k`-fold, the data is randomly split into `k` subsets, or folds, which can result in an uneven distribution of class labels in each fold, particularly in imbalanced datasets. Stratified `k`-fold addresses this by ensuring that each fold has a representative balance of classes, similar to the overall dataset.

- Stratified group `k`-fold cross-validation is an extension of stratified `k`-fold designed for scenarios where data is grouped into clusters or subsets. It combines stratification, ensuring that each fold maintains the class distribution, with group partitioning, ensuring that all data from a particular group appears in only one fold.

For a detailed and graphical explanation of both approaches, we can also consult the excellent `scikit-learn` [user guide](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-with-stratification-based-on-class-labels), which this implementation is also based on.

## Custom Models

If we want to utilize models not currently implemented in the package, we can inject the custom model implementations via two approaches.
The first approach is to directly modify the two core model files, e.g. `woodnet.models.planar` or
`woodnet.models.volumetric` such that they contain our new model implementation. This allows direct instantiation via the YAML-configuration file workflow. A drawback would be, that git merge conflicts might arise when pulling new updates from the remote repository. Also, poorer code structuring due to mixing of origins/concerns would be in effect.
The second option works by copying your implementation file into the `woodnet.models` submodule of the full package.
In practice, we can just put our custom model implementation inside a separate Python module (i.e. a `.py` file).
> [!IMPORTANT]
> The file should use an appropriate name with the indication prefix `customcontrib_$DESIREDNAME.py`, where the prefix with the trailing underscore `customcontrib_` must be used exactly.
Then, we can copy this module to the `woodnet.models` submodule and use the custom model via the YAML configuration file workflow.

If we create a custom model `SuperModel` in the file `customcontrib_SuperModel.py` like so:

```python
import torch

class SuperModel(torch.nn.Module):
    # supermodel implementation here
    pass
```

then the file should be placed like so in the cloned repository layout:

```
.
└── woodnet/
    └── src/
        └── woodnet/
            └── models/
                ├── volumetric.py                    # example builtin model file
                └── customcontrib_SuperModel.py      # user-supplied model module
```

Currently, this is somewhat brittle and inconvenient. We do plan to improve this.
The custom model implementation modules are then collected via a filename matching scheme and are available for the name-based instantiation logic.
Note that when we create models from the configuration, the first model class with a matching name is used. If we implement custom models with the same name as already implemented model, name shadowing
may lead to errors. Thusly pick an unique model class name.

## Miscellaneous

### Further Usage Ideas

The presented pipeline implementation could serve in different to the wood science community.
Firstly, the implementation could be adopted as a purpose-built template to inject **custom CT data** of wood samples to gauge classification performance for this specific dataset.
Furthermore, adoption to **light microscopic** datasets is easily conceivable since a fully planar 2D formulation is included in the package.
Also, usage with **multiplanar microscopic** images is possible. For this, the triaxial formulation with a preset ordering for the typical wood anatomic cross sections may be appropriate.

### Bugs, Questions, Requests and Contributions

If you find bugs or have general questions please do not hesitate to open an issue. We will gladly try to answer and improve the pipeline.
Also, we would be happy to include feature requests or use cases if they are within the general scope of our pipeline. For this, also head over to the repository issues tab and open with label `enhancement`! 🧰

### Acknowledgements

See e.g. our citations and literature in the paper manuscript.
Also many thanks to the incredible global open-source community and its fruits of labor ❤️ i.e. the dependencies crucial to this implementation:
Python, PyTorch, monai, pytest, sphinx, and many more!

## About

Author Jannik Stebani. Released under the MIT license.
Accompanying manuscript: TODO:INSERT by Jannik Stebani, Tim Lewandrowski, Kilian Dremel, Simon Zabler and Volker Haag