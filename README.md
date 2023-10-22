# Small Kernels

## Purpose
This quick exercise was just to familiarize myself with writing CUDA kernels without having to worry too much about external factors and focus on the actual kernel construction itself. Some of the kernels build include filtering through arrays or summing up chunks of an array in a parallel manner.

## Setup
If you wish to run these kernels for yourself or go through the exercies, just run these instruction:

```bash
ENV_PATH=./small_kernels/.env/
conda create -p $ENV_PATH python=3.9 -y
conda install -p $ENV_PATH pytorch=2.0.0 torchtext torchdata torchvision -c pytorch -y
conda install -c conda-forge cudatoolkit-dev
conda install pycuda
conda run -p $ENV_PATH pip install -r requirements.txt
```

If you are on Windows, you can run this:

```bash
$env:ENV_PATH='c:\users\<user_name>\small_kernels\.env'
conda create -p $env:ENV_PATH python=3.9 -y
conda install -p $env:ENV_PATH pytorch=2.0.0 torchtext torchdata torchvision -c pytorch -y
conda install -c conda-forge cudatoolkit-dev
conda install pycuda
conda run -p $ENV_PATH pip install -r requirements.txt
```

## Acknowledgements

Much of this implementation was guided by a program created by Redwood Research. Many thanks to Redwood for creating this program and serving as a stepping stone for this implenentation.
