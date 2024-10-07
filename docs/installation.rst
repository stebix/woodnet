Installation
============

To install the project into your local machine, first pull the repository to a location you desire.

.. code-block:: bash

    cd /desired/location
    git pull
    git pull https://github.com/stebix/woodnet.git

We recommend to install the package using some sort of virtual environment. For heavy deep-learning machinery with compiled/C/C++/CUDA components like PyTorch,
we recommend using `conda` or the modern and faster sibling `mamba`. If you decide to use `mamba`, supplant the `conda` command with `mamba`.

As a first step, create the environment from the provided `environment.yaml` file in the repository. This will preinstall all necessary dependencies.
With the current working directory being the pulled repository, we can execute

.. code-block:: bash

    conda env create -f environment.yaml


The default name for the newly created environment is `woodanalysis`. If we want another environment name, we can just modify the first line of the environment file.
We than need to activate the environment and may inspect the correct installation of the required packages can be inspected via the `list` command.

.. code-block:: bash

    conda activate woodanalysis
    conda env list

Of course, modifications to the environment name have to be respected here.
Then, we can install a local editable version of the package via `pip` using the command

.. code-block:: bash

    pip install --editable .


Then the package is importable as any Python package for sessions started within this environment 🎉
This installation process also allows you to use the package or parts of it as a (modifiable) library in the context different from the current use cases.

.. note::
    For future releases, we plan to switch to a full PyPI and `anaconda` package release. But currently, the above clone + install method is the recommended one!
    Any updates can be retrieved by navigating to the repository and pulling in the changes. Via the editable install, these are then directly available environment-wide.