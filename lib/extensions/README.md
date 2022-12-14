### extensions
* `mesh2sdf_cuda`: a fast SDF CUDA kernel for meshes based on [DualSDF](https://github.com/zekunhao1995/DualSDF)'s implementation

### Build

For all extensions, a `setup.py` file is provided for easy building. Build as follows:
```bash
cd <extension_name> && python3 setup.py install --user
```
You might need a specific version of GCC to make this work, e.g. `gcc-8/g++-8` as CUDA 10.1+ is not supported for more recent versions. The above will install all extensions _globally_ so they can be imported from anywhere in the project. This is important for the future Docker image for NGC support. This directory is mostly to have a single directory for _all_ C++ SDF-related extensions in the future.
