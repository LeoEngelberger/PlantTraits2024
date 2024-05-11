# Project For PlanTraits2024 Competition
  this is a project that tries to create a NN-model with Keras and PyTorch

## Description
This is my personal attempt at the PlantTraits 2024 challenge. It aims to make the most of Cuda on a Windows System. It usese Torch as a Backend for keras. This allows it to interact with the gpu without having to involve wsl2
The Performance seems to be nicely outsourced to the GPU with this model.


To unfold full potential the project should run in a conda environment that supports cuda
  run the following commands in your activated conda environment:

conda install pytorch torchvision torchaudio pytorch-cuda=[YOUR CUDA VERSION e.g. 12.1] -c pytorch -c nvidia
conda install keras -c conda-forge
pip uninstall h5py
pip install h5py

(maybe you dont need the last to pip calls but i needed those to get everything running.)

Get Dataset of kaggle Project (open starternotebook in colab, copy the second cell into a .py file. run said file done.)

run main.py
