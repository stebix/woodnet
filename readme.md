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
  softwood : 0
  hardwood : 1
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
specied and for both classes we have a large number of samples. For both classes, we obtained
samples from freshly logged wood that we mark with the `group: pristine` attribute.
We additionally got samples that were exposed to the elements and mark these with the
`group: withered` attribute. We can use the `group` data instance attribute during the comutation of the cross-validation splits of the specified instances into the training set and the validation set.
In addition to the "default" variant of class-stratified `k`-fold cross-validation we may then
employ group-wise `k`-fold cross-validation. Then we can evaluate whether the model is able/flexible/intelligent enough to generalize across groups.


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

- `tensorboard` log file: We use this library to visualize and analyse the training experiment on the fly. More on this later. This file is also located in the `logs` directory.

The directory will be created if it is not present. Due to the unqiueness of all above artifacts to a single training experiments it is highly recommended to choose a new training directory for each individiual training experiment.


#### Device

The device option lets us choose the device on which we want to perform the training experiment calculation. The common options are `cpu` for (often infeasibly slow) central processing unit (CPU) training or `cuda` for accelerated graphic processing unit (GPU) training. For systems that sport multiple GPUs, we can use `cuda:$N` with `$N` indicating an appropriate integer that pins the specific GPU in our system on which we desire the traniing experiment to run on.


### Model Block

In the model block, we configure our core deep learning model.
In genreal, we can set all user-facing parameters in the initializer (i.e. `__init__` method) of the model class here. Additionally, model ahead-of-time (AOT) and just-in-time (JIT) compilation
flags can be set here. For more information on AOT and JIT-functionality via `torch.compile` please consider the [PyTorch docs](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).
A typical model block may look like this:
```yaml
model:
  name: ResNet3D
  in_channels: 1
  compile:
    enabled: True
    dynamic: False
    fullgraph: False
```
In this example, we selected the `ResNet3D` from our model zoo and configured it to have a single input channel. Single channel data is typical for monochromatic computed tomography data. For light microscopy data, we may encounter multi channel data due to the separate measurement of red, green and blue intensity (RGB) in a photographic sensor.
We also set the model compilation flag. Thusly, the model will be compiled at the first iteration at the cost of a small, singular latency increase and the benefit of substantial acceleration during following iterations.


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
Again, the loss function class is selected via the `name` field that must mathc the desired loss function class of Pytorch.
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
```
For a validation run, the training is paused and predictions for all validation data instances will be performed. The result of this run (i.e. the validation metric score) is reported to the log file and sent to the tensorboard inspection tool.
Also, the model weights are saved as a checkpoint if the score for a validation run is optimal or in the top-`k`-optimal range.
We can set the maximum number of iterations and epochs as an exit condition for conclusion of the training experiment. Note that the system exits the experiment run as soon as the first of both criterions is fulfilled.


## About

Author Jannik Stebani. Released under the MIT license.
Accompanying manuscript: TODO:INSERT