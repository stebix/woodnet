Usage
=====

``net`` CLI
------------

The ``woodnet`` can be utilized as both a command line tool and a Python package.
To invoke the CLI for training, prediction, evaluation and selected global
configuration tasks, use the ``net`` commend. Then, different action verbs can be
used to specify the desired action.
Here is a list of the available action verbs:

- ``train`` : Run a single training experiment configured via a YAML file.
- ``batchtrain`` : Run multiple training experiments sequentially (multiple YAML files)
- ``predict`` : Run a single prediction task configured via a YAML file.
- ``evaluate`` : Evaluate and aggregate over multiple training experiments in the context of cross validation.

For the core of the verbs, i.e. ``train``, ``batchtrain`` and ``predict``, simply a configuration file is required.
The configuration file can be indicated via a path-like string following the verb.
For exmaple, to train a model with a configuration file located at ``/path/to/config.yaml``, the following command can be used:

.. code-block:: bash

    net train /path/to/config.yaml

Similarly, prediction and batchtraining can be invoked with the respective configuration files.
The evaluation verb is a bit more complex, please refer to the :ref:`detailed documentation<evaluate_cli_doc>` below.
The full list of available options can be retrieved by using the ``--help`` flag with the `net` command.
Below are the detailed documentation for each of the action verbs.

.. _train_cli_doc:

.. sphinx_argparse_cli:: 
    :module: woodnet.parsing
    :func: _make_train_parser
    :prog: net train
    :description: Run a single training experiment configured via a YAML file located on the file system.
    :title: net train

.. _batchtrain_cli_doc:

.. sphinx_argparse_cli:: 
    :module: woodnet.parsing
    :func: _make_batchtrain_parser
    :prog: net batchtrain
    :description: Run multiple training experiments sequentially using multiple YAML files located on the file system.
    :title: net batchtrain

.. _predict_cli_doc:

.. sphinx_argparse_cli:: 
    :module: woodnet.parsing
    :func: _make_predict_parser
    :prog: net predict
    :description: Run a single prediction task configured via a YAML file located on the file system.
    :title: net predict

.. _evaluate_cli_doc:

.. sphinx_argparse_cli::
    :module: woodnet.parsing
    :func: _make_evaluate_parser
    :prog: net evaluate
    :description: Evaluate and aggregate over multiple training experiments in the context of cross validation.
    :title: net evaluate


Auxiliary CLI
----------------

``refolder`` CLI
^^^^^^^^^^^^^^^^

The refolder CLI is a tool to modify existing training configuration file with
a cross validation context in mind.
For a cross validation experiment, one can use a single template configuration file
and create a CV-specific new configuration by splitting/rearranging of dataset instances into
the training and validation sets.
This task can be done with the ``refolder`` command.

.. _refolder_cli_doc:

.. sphinx_argparse_cli:: 
    :module: woodnet.refolder
    :func: create_parser
    :prog: refolder
    :description: Create a new cross-validation-wise training configuration from an existing training configuration template.
    :title: refolder


``benchmark`` CLI
^^^^^^^^^^^^^^^^^

The benchmark CLI is a tool to run a benchmarking the performance of 
the processing pipeline for a given hardware infrastructure.

.. _benchmark_cli_doc:

.. sphinx_argparse_cli:: 
    :module: woodnet.benchmark
    :func: create_parser
    :prog: benchmark
    :description: Run a benchmarking experiment to evaluate the performance of the processing pipeline for various parameter settings.
    :title: benchmark