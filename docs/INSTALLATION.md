# INSTALLATION

<!-- these installation .sh files are old!! -->
<!-- To install the dependencies execute the `scripts/install_dependencies.sh`

```bash
scripts/install_conda.sh # Skip this step if you already have conda installed.
scripts/install_dependencies.sh
```

Please see the [README](README.md) for a quick start instruction. -->


<!-- Alternatively,  -->
you can follow the detailed instructions to setup the software from scratch

#### 1. Environment

Install miniconda if not already present on the current system.You can use `scripts/install_conda.sh` for this step:
```bash
sudo apt install curl 

curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh

SHELL_NAME=`basename $SHELL`
eval "$($HOME/miniconda3/bin/conda shell.${SHELL_NAME} hook)"
conda init ${SHELL_NAME}
conda install mamba -c conda-forge
conda config --set auto_activate_base false
```

Next, create the rlbench environment and install the dependencies

```bash
conda create -n rlbench python=3.8
conda activate rlbench
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 2. PyRep and Coppelia Simulator

Follow instructions from the [PyRep fork](https://github.com/markusgrotz/PyRep); reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd third_party
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install -e .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

#### 3. RLBench

PerAct^2 uses the [RLBench fork](https://github.com/markusgrotz/RLBench). 

```bash
cd third_party
cd RLBench
pip install -r requirements.txt
pip install -e .
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).

#### 4. YARR

PerAct² uses the [YARR fork](https://github.com/markusgrotz/YARR).

Before installation, downgrade pip to avoid compatibility issues
with older `omegaconf` versions required by `hydra-core==1.0.5`.

```bash
python -m pip install "pip<24.1"
pip install setuptools==61.1.0
```

Then install YARR in editable mode:

```bash
cd third_party/YARR
pip install -e .
```

If dependency resolution fails, manually install compatible versions:

```bash
pip install "omegaconf>=2.0.5,<2.1"
pip install importlib-resources
pip install fsspec
```

You can verify the installation with:

```bash
python -c "import yarr, hydra, omegaconf; print(hydra.__version__, omegaconf.__version__)"
```

#### 5. pytorch3d

PerAct² uses a local pytorch3d fork.

Install required dependencies first:

```bash
cd third_party/pytorch3d

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

Before installation, make sure your CUDA toolkit version matches
the CUDA version used to compile PyTorch.

You can check with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
```

For example:

- PyTorch CUDA 11.8 ↔ nvcc 11.8
- PyTorch CUDA 12.1 ↔ nvcc 12.1

If these versions mismatch, pytorch3d compilation will fail with:

```text
RuntimeError: The detected CUDA version mismatches the version that was used to compile PyTorch
```

(Optional but recommended)

```bash
conda install ninja
```

Then install pytorch3d in editable mode:

```bash
pip install -e .
```

Note:
- pytorch3d compiles CUDA/C++ extensions during installation.
- Compilation may take several minutes.
- Processes such as `cc1plus`, `cudafe++`, or `nvcc` are expected during build.

You can verify the installation with:

```bash
python -c "import pytorch3d; print(pytorch3d.__version__)"
python -c "import torch; import pytorch3d._C; print('pytorch3d C extension ok')"
```
#### 6. wandb

Install the compatible wandb version used by PerAct²:

```bash
pip install wandb==0.14.0
```

wandb 0.14.0 may downgrade or modify some dependencies
(e.g. protobuf), which can trigger additional missing dependency
warnings for rendering or video-related packages.

If pyrender-related dependency warnings appear, install:

```bash
pip install freetype-py imageio "pyglet>=1.4.10" PyOpenGL==3.1.0 scipy trimesh
```

For movie/video utilities compatibility with older research codebases:

```bash
pip install "moviepy<2.0"
pip install "decorator<6.0,>=4.0.2" imageio_ffmpeg "proglog<=1.0.0" python-dotenv
```

We recommend keeping:

- `pip < 24.1`
- `numpy == 1.24.x`
- `moviepy == 1.0.3`

to avoid compatibility issues with older Hydra/OmegaConf and
PerAct² dependencies.

You can verify the installation with:

```bash
python -c "import wandb; print(wandb.__version__)"
python -c "import wandb, pyrender, moviepy; print('wandb/pyrender/moviepy ok')"
```
