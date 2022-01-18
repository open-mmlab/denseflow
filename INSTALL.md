# Install Guide

## Dependencies:

- CUDA (driver version > 400)
- OpenCV v3/v4
- Boost
- HDF5 (Optional)

### 1. Install CUDA

Make sure that the driver version is >400.
Otherwise, the speed would be painfully slow, although it compiles and runs.

### 2. Install OpenCV with CUDA support

We need OpenCV with CUDA enabled.
An install script can be found [Here](https://github.com/innerlee/setup/blob/master/zzopencv.sh).
Since OpenCV itself has many dependencies,
you can follow the example below to compile.

```bash
# ZZROOT is the root dir of all the installation
# you may put these lines into your `.bashrc` or `.zshrc`.
export ZZROOT=$HOME/app
export PATH=$ZZROOT/bin:$PATH
export LD_LIBRARY_PATH=$ZZROOT/lib:$ZZROOT/lib64:$LD_LIBRARY_PATH

# fetch install scripts
git clone https://github.com/innerlee/setup.git
cd setup

# opencv depends on ffmpeg for video decoding
# ffmpeg depends on nasm, yasm, libx264, libx265, libvpx
./zznasm.sh
./zzyasm.sh
./zzlibx264.sh
./zzlibx265.sh
./zzlibvpx.sh
# finally install ffmpeg
./zzffmpeg.sh

# install opencv 4.3.0
./zzopencv.sh
# you may put this line into your .bashrc
export OpenCV_DIR=$ZZROOT

```

### 3. Install Boost Library

```bash
# install boost
./zzboost.sh
# you may put this line into your .bashrc
export BOOST_ROOT=$ZZROOT
```

### 4. Install HDF5 Library (Optional)

```bash
# install hdf5
./zzhdf5.sh
```

### 5. Install denseflow

```bash
# finally, install denseflow
./zzdenseflow.sh
```
