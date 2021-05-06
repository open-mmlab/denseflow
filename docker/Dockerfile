ARG CUDA="11.1.1"
ARG CUDNN="8"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

ARG OPENCV="4.5.2"
# "Pascal" "Volta" "Turing" "Ampere"
ARG CUDA_GENERATION="Pascal"

RUN apt update \
    && apt install -y git cmake wget software-properties-common nasm yasm libx264-dev libx265-dev libvpx-dev libboost-all-dev ffmpeg libavcodec-dev libavformat-dev libavutil-dev libblas-dev liblapack-dev libswscale-dev libtiff-dev libdc1394-22-dev libpng-dev libavresample-dev ccache libgflags-dev libhdf5-dev liblapack-dev libeigen3-dev libgoogle-glog-dev libfreetype6-dev \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
    && apt update \
    && apt install cmake -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt /tmp/* \
    && wget https://github.com/opencv/opencv/archive/${OPENCV}.tar.gz -O opencv.tar.gz \
    && wget https://github.com/opencv/opencv_contrib/archive/${OPENCV}.tar.gz -O opencv_contrib.tar.gz \
    && mkdir opencv \
    && mkdir opencv_contrib \
    && tar xf opencv.tar.gz -C opencv/ --strip-components 1 \
    && tar xf opencv_contrib.tar.gz -C opencv_contrib/ --strip-components 1 \
    && cd opencv \
    && mkdir build \
    && cd build \
    && cmake \
        -DBUILD_EXAMPLES=OFF \
        -DWITH_QT=OFF \
        -DCUDA_GENERATION=${CUDA_GENERATION} \
        -DOpenGL_GL_PREFERENCE=GLVND \
        -DBUILD_opencv_hdf=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DBUILD_opencv_cnn_3dobj=OFF \
        -DBUILD_opencv_dnn=OFF \
        -DBUILD_opencv_datasets=OFF \
        -DBUILD_opencv_aruco=OFF \
        -DBUILD_opencv_tracking=OFF \
        -DBUILD_opencv_text=OFF \
        -DBUILD_opencv_stereo=OFF \
        -DBUILD_opencv_saliency=OFF \
        -DBUILD_opencv_rgbd=OFF \
        -DBUILD_opencv_reg=OFF \
        -DBUILD_opencv_ovis=OFF \
        -DBUILD_opencv_matlab=OFF \
        -DBUILD_opencv_freetype=OFF \
        -DBUILD_opencv_dpm=OFF \
        -DBUILD_opencv_face=OFF \
        -DBUILD_opencv_dnn_superres=OFF \
        -DBUILD_opencv_dnn_objdetect=OFF \
        -DBUILD_opencv_bgsegm=OFF \
        -DBUILD_opencv_cvv=OFF \
        -DBUILD_opencv_ccalib=OFF \
        -DBUILD_opencv_bioinspired=OFF \
        -DBUILD_opencv_dnn_modern=OFF \
        -DBUILD_opencv_dnns_easily_fooled=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_NEW_PYTHON_SUPPORT=ON \
        -DBUILD_opencv_python3=OFF \
        -DHAVE_opencv_python3=OFF \
        -DPYTHON_DEFAULT_EXECUTABLE="$(which python)" \
        -DWITH_OPENGL=ON \
        -DWITH_VTK=OFF \
        -DFORCE_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_GDAL=ON \
        -DCUDA_FAST_MATH=ON \
        -DWITH_CUBLAS=ON \
        -DWITH_MKL=ON \
        -DMKL_USE_MULTITHREAD=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DWITH_CUDA=ON \
        -DNVCC_FLAGS_EXTRA="--default-stream per-thread" \
        -DWITH_NVCUVID=OFF \
        -DBUILD_opencv_cudacodec=OFF \
        -DMKL_WITH_TBB=ON \
        -DWITH_FFMPEG=ON \
        -DMKL_WITH_OPENMP=ON \
        -DWITH_XINE=ON \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DCMAKE_INSTALL_PREFIX=/usr \
        .. \
    && make -j"$(nproc)" \
    && make install \
    && cd ../../ \
    && git clone https://github.com/open-mmlab/denseflow \
    && cd denseflow \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr -DUSE_HDF5=yes .. \
    && make -j"$(nproc)" \
    && make install \
    && cd ../../ \
    && rm -rf *gz open* dense*

ENTRYPOINT ["/usr/bin/denseflow"]
