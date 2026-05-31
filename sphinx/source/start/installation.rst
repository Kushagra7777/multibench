Installation
************

Prerequisites
=============

- Python >= 3.10
- PyTorch (CPU or GPU)

Clone the repository first:

.. code-block:: bash

   git clone https://github.com/pliang279/MultiBench.git
   cd MultiBench

Windows users should check out the Windows Subsystem for Linux (WSL).

Virtual environment
===================

It is advised to use a virtual environment to manage dependencies. You can
create one using ``uv`` (recommended) or the built-in ``venv`` module.

**Using uv (recommended):**

.. code-block:: bash

   uv venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate

**Using venv:**

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate

If you are not using ``uv``, simply drop the ``uv`` prefix from the
installation commands below (e.g. ``pip install -r requirements.txt``).

Installing dependencies
=======================

**GPU support.** Install PyTorch using the appropriate command from the
`PyTorch website <https://pytorch.org/get-started/locally/>`_. The following
installs the latest stable PyTorch with CUDA support, then the remaining
requirements:

.. code-block:: bash

   uv pip install -r requirements.txt

**CPU only.** Install the CPU build of PyTorch directly:

.. code-block:: bash

   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   uv pip install memory-profiler scikit-learn scipy matplotlib h5py tqdm

**Conda (alternative).** An ``environment.yml`` is also provided:

.. code-block:: bash

   conda env create [-n ENVNAME] -f environment.yml

From there, you should be able to try out the example scripts and the rest of
the code by running a Python kernel inside the repository folder. Head to the
:doc:`quickstart` guide for a runnable first experiment.
