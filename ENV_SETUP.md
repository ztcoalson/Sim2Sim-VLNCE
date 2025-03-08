# Conda environment set-up for Sim2Sim-VLNCE.

These custom set-up instructions helped me solve two errors I encountered when following the original set-up guide:
(1) `headless` not being recognized by `habitat-sim` and (2) not being able to install `python-opencv`. The fixes, detailed below, are to build `habitat-sim` with `headless` from source and install a specific version of `python-opencv`.

1. Create the conda environment with the preliminary dependencies:

```bash
conda create -n sim2sim python=3.6 caffe-gpu
conda activate sim2sim
```

2. Install the dependencies for `habitat-sim=0.1.7`:

```bash
conda install --only-deps habitat-sim=0.1.7 -c aihabitat
```

3. When I installed `habitat-sim=0.1.7` directly from conda, `headless` did not seem to be configured properly. To solve this, we can install `habitat-sim=0.1.7` (with `headless`) from source:

```bash
wget https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
conda install --offline habitat-sim-0.1.7-py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
rm -f habitat-sim-0.1.7-py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2   # optional
```

4. Before installing pip dependencies, we need to change instances of `python-opencv>=3.3.0` to `opencv-python==3.4.0.12`. You need to do this in `./requirements.txt` and `./habitat-lab/requirements.txt`. Without this, pip tries to install later versions of the package that require wheels, which take forever to build and in my case resulted in compilation errors.

5. Install the remaining pip dependencies:

```bash
pip install -r habitat-lab/requirements.txt
pip install -r habitat-lab/habitat_baselines/rl/requirements.txt
pip install -r habitat-lab/habitat_baselines/rl/ddppo/requirements.txt
pip install -r habitat-lab/habitat_baselines/il/requirements.txt
pip install -r requirements.txt
pip install -e habitat-lab/
pip install -e transformers/
```