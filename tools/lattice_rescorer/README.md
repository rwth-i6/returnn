How to use the lattice rescorer tool.



# Install TensorFlow with pip.

    $ pip3 install --user --upgrade tensorflow-gpu
    
Verify the install:
    
    $ python3 -c "import tensorflow as tf; print(tf.__version__)"

If it doesn't work, please read https://www.tensorflow.org/install/pip

# Compile TensorFlow c++ library from source

## Prepare environment for Linux
    
  Before compiling Tensorflow c++ library on Linux, install the following build tools on your system:
  *bazel
  *TensorFlow Python dependencies
  *optionally, NVIDIA packages to support TensorFlow for GPU.
    
### Install Bazel

Refer to https://docs.bazel.build/versions/master/install-ubuntu.html#installing-using-binary-installer)


### Install Tensorflow Python dependencies

To install TensorFlow, you must install the following packages:numpy, dev, pip, wheel

### Install TensorFlow for GPU prerequisites

The following NVIDIA hardware must be installed on your system:
GPU card with CUDA Compute Capability 3.0 or higher. See NVIDIA documentation for a list of 
supported GPU cards.
The following NVIDIA software must be installed on your system:
CUDA Toolkit (>= 7.0). We recommend version 9.0. For details, see NVIDIA's documentation. 
Ensure that you append the relevant CUDA pathnames to the LD_LIBRARY_PATH environment 
variable as described in the NVIDIA documentation.
GPU drivers supporting your version of the CUDA Toolkit.
cuDNN SDK (>= v3). We recommend version 7.0. For details, see NVIDIA's documentation.

#### Install CUDA and cuDNN

Note: tensorflow c++ library requires libcudnn.so.7. If you have to install cuDNN 7 locally:
download the cudnn 7.0.5 for cuda 9.0 from https://developer.nvidia.com/rdp/cudnn-archive
Then execute the following commands:

    $ tar -xzvf cudnn-9.0-linux-x64-v7.tgz
    $ mv cuda cudnn-9.0-v7.0
    
#### Set up your environment

Add the following command to .bashrc:

    export LD_LIBRARY_PATH=/path-to-your-cudnn/lib64:LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 

## Clone the Tensorflow repository and configure the installation

    $ git clone https://github.com/tensorflow/tensorflow
    $ cd tensorflow
    $ ./configure
    
## Compile Tensorflow c++ libraries

    $ export TEST_TMPDIR=/u/username/bazel_outputRoot
    $ tensorflow/contrib/makefile/download_dependencies.sh
    $ bazel build -c opt --config=cuda --local_resources=6144,4,1.0 --jobs=4 //tensorflow:libtensorflow_cc.so
    $ bazel build -c opt --config=cuda --local_resources=6144,4,1.0 --jobs=4 //tensorflow:libtensorflow.so 
    
The compiled libraries are stored in /u/username/tensorflow/bazel-bin/tensorflow

## Set up your environment

Add the following command to .bashrc:

    export LD_LIBRARY_PATH=/u/username/tensorflow/bazel-bin/tensorflow:LD_LIBRARY_PATH

# Compile LSTM Op in RETURNN

For example:

    $ ./path-to-your-RETURNN/tools/compile_native_op.py --native_op LstmGenericBase --output xyz

xyz contains the paths to the compiled libraries. The libraries are stored by default in /var/tmp/, please move the libraries to somewhere else.

# Create the Tensorflow graph for forwarding

    $ ./path-to-your-RETURNN/tools/compile_tf_graph.py <filename to config-file> --eval 1 --output_file <filename to output pb or pbtxt file>

# Example to use lattice rescorer

## Makefile

Before using the Makefile, please verify the paths to your Tensorflow source code, cuda and cuDNN in Makefile

## Gernerate executable file from source files

In the folder returnn/tools/lattice_rescorer, execute the following command

    $ make
    
A executable file named lattice_rescorer will be generated in the same folder.

## An example script to rescore a lattice

The script example/rescore_lattice.sh is an example to rescore a lattice.

Before using the script, please modify manually the files libs_list, state_vars_list, tensor_names_list, and then

    $ ./rescore_lattice.sh
