# To run the algorithm on personal computer:

## python version:

  - Python: 3.12

## Libraries

conda install pytorch torchvision torchaudio -c pytorch

conda install -c conda-forge nibabel numpy scipy scikit-sparse dipy petsc petsc4py

## How to run the code

1) Prepare the data, only the raw data MultiTE_dMRI is needed
2) Run Converting.ipynb. This will save the processed data
3) Run Analyzing.ipynb 


# To run the algorithm on HPC:

## Conda version:

  - conda 24.11.1

## python version:

  - Python: 3.12

## Libraries

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy cuda-version=11.8

conda install -c conda-forge nibabel numpy scipy scikit-sparse dipy petsc petsc4py

## How to run the code

1) Upload the processed data from personal computer to HPC
2) Run HPC_Pre_Process.py from RunME.sh
3) Run HPC_Post_Process.py from RunME.sh
4) Download the processed data from HPC to personal computer
5) Run ReadingHPC.ipynb to read the results



