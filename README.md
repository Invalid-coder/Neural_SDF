### Python dependencies
The easiest way to get started is to create a virtual Python 3.8 environment:
```
conda create -n my_env python=3.8
conda activate my_env
pip install --upgrade pip
pip install -r requirements.txt
```

The code also relies on [OpenEXR](https://www.openexr.com/), which requires a system library:

```
sudo apt install libopenexr-dev 
pip install pyexr
```

To see the full list of dependencies, see the [requirements](requirements.txt).

### Building CUDA extensions
To build the corresponding CUDA kernels, run:
```
cd sdf-net/lib/extensions
cd <extension_name> && python3 setup.py install --user
```

The above instructions were tested on Ubuntu 18.04/20.04 with CUDA 10.2/11.1.

## Training

**Note.** All following commands should be ran within the `sdf` directory.

### Download sample data

To download a cool armadillo:

```
wget https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj -P data/
```

To download a cool matcap file:

```
wget https://raw.githubusercontent.com/nidorx/matcaps/master/1024/6E8C48_B8CDA7_344018_A8BC94.png -O data/matcap/green.png
```

### Training from scratch

```
python app/main.py \
    --net OctreeSDF \
    --num-lods 5 \
    --dataset-path data/armadillo.obj \
    --epoch 250 \
    --exp-name armadillo
```

This will populate `_results` with TensorBoard logs.

