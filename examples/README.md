# Multibench Example  

## Local Python files:  
- Multibench_Example_Usage_Part_0_UNI.py
- Multibench_Example_Usage_Part_1_MOSI.py
- Multibench_Example_Usage_Part_2_MFAS.py
- Multibench_Example_Usage_Part_3_MCTN.py

## Local Jupyter notebooks:  

### Local requirements:  

- install the requirements in `requirements.txt` using `uv pip install -r requirements.txt` in the terminal.
- Download dataset (e.g. MOSI) and place it in the respective folder (e.g. `data/affect`).
- Fow local jupyter notebooks, you need to do the following:  
 - install ipykernel using `uv pip install ipykernel` in the terminal.
 - run `uv ipython kernel install --user --name=multibench` in the terminal to create a new kernel named `multibench`.
 - open the jupyter notebook and change the kernel to `multibench` (you can do this by clicking on "Kernel" in the top menu, then "Change kernel", and then selecting "multibench" from the list).
 - If it doesn't work, run **Developer: Reload Window** in the command palette (you can open the command palette by pressing Ctrl+Shift+P or Cmd+Shift+P on Mac, then typing "Developer: Reload Window" and pressing Enter).

## Colab Links to the Tutorials:

- [Tutorial 1](https://colab.research.google.com/drive/1PfgotUdIWQvA0mIUbncLPSRzqAPXKZV9): This example shows a very basic usage case of MultiBench. In particular, it demonstrates how to use MultiBench with the affective computing dataset MOSI, and how to use it with a very simple fusion model.
- [Tutorial 2](https://colab.research.google.com/drive/1ywJ06rC5WRMEGrCtWE5g9DWdv3U9p8xa): This example shows a slightly more complicated training paradigm in MultiBench - MFAS (MultiModal Fusion Architecture Search) model on the AVMNIST dataset.
- [Tutorial 3](https://colab.research.google.com/drive/1M76QyqdaqkaKU786HnM4qxxnkxMAcex5): This example shows a slightly more complicated training paradigm in MultiBench. Namely, we'll run MCTN (learning representations by translating from one modality to another) on MOSI.

