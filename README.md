# deformgs-reloaded

## Installation

### Container Setup

```bash
docker build -t deformgs-reloaded .
docker run -dit --name deformgs_reloaded --gpus all --network=host --shm-size=50G deformgs_reloaded
```

### Inside the container
```bash
git clone https://github.com/JanIsHacking/deformgs-reloaded.git
cd deformgs-reloaded
git submodule update --init --recursive
conda create -n deformgs-reloaded python=3.7
conda activate deformgs-reloaded
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
```

### Training

```bash
python train.py
```
