How to use the lattice rescorer tool.
=====================================

**Tested version infos:**    
tensorflow-gpu pip package 1.8.0   
tensorflow source code 1.7.0   
bazel 0.11.0  
protoc 3.2.0  
cuda 9.0  
cuDNN 7.0  
g++ 5.4.0  

# Install TensorFlow with pip.

[install tensorflow with pip](https://www.tensorflow.org/install/pip)

    $ pip3 install --user --upgrade tensorflow-gpu
    
Verify the install:
    
    $ python3 -c "import tensorflow as tf; print(tf.__version__)"

# Compile TensorFlow c++/c library from source

You can refer to [install tensorflow from source](https://www.tensorflow.org/install/source),the steps are similar.  
Also, the scirpt in Kaldi [install_tensorflow_cc.sh](https://github.com/kaldi-asr/kaldi/blob/master/tools/extras/install_tensorflow_cc.sh) shows the steps to compile c++/c libraries.

## Prepare environment for Linux
    
  Before compiling Tensorflow c++ library on Linux, install the following build tools on your system:  
  *bazel  
  *cuda, cuDNN  
    
### Install Bazel

You should install a required version of bazel, [version information](https://www.tensorflow.org/install/source#tested_build_configurations).  
[Install bazel using binary installer](https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu)

#### Download Bazel

Download the Bazel binary installer named bazel-<version>-installer-linux-x86_64.sh from the [Bazel releases page on GitHub](https://github.com/bazelbuild/bazel/releases).

#### Run the installer

Run the Bazel installer as follows:

    $ chmod +x bazel-<version>-installer-linux-x86_64.sh
    $ ./bazel-<version>-installer-linux-x86_64.sh --user

#### Set up your environment

In your .bashrc, add the following line:
    
    export PATH="$PATH:$HOME/bin"

### Install TensorFlow for GPU prerequisites

#### Install CUDA and cuDNN

You should install a required version of CUDA and cuDNN, [version information](https://www.tensorflow.org/install/source#tested_build_configurations):  

[Install CUDA](https://developer.nvidia.com/cuda-toolkit-archive)  

[Install cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)  

Then execute the following commands:

    $ tar -xzvf cudnn-9.0-linux-x64-v7.tgz
    $ mv cuda cudnn-9.0-v7.0
    
#### Set up your environment

Add the following command to .bashrc:

    export LD_LIBRARY_PATH=/path-to-your-cudnn/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/path-to-your-cuda/extras/CUPTI/lib64 

## Clone the Tensorflow repository and configure the installation

You should download a proper version of TensorFlow source code, [version information](https://www.tensorflow.org/install/source#tested_build_configurations)  
Use Git to clone the TensorFlow repository

    $ git clone https://github.com/tensorflow/tensorflow
    $ cd tensorflow
    
The repo defaults to the master development branch. You can also checkout a release branch to build:

    $ git checkout branch_name  # r1.9, r1.10, etc.
    $ ./configure
    
This script prompts you for the location of TensorFlow dependencies and asks for additional build configuration options (compiler flags, for example).
[A sample configuration session](https://www.tensorflow.org/install/source#configure_the_build)

## Compile Tensorflow c++ libraries

    $ export TEST_TMPDIR=/u/username/bazel_outputRoot
    $ tensorflow/contrib/makefile/download_dependencies.sh 
    $ bazel build -c opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --local_resources=6144,4,1.0 --jobs=4 //tensorflow:libtensorflow_cc.so
    $ bazel build -c opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --local_resources=6144,4,1.0 --jobs=4 //tensorflow:libtensorflow.so 
    
The compiled libraries are stored in /u/username/tensorflow/bazel-bin/tensorflow

## Set up your environment

Add the following command to .bashrc:

    export LD_LIBRARY_PATH=/u/username/tensorflow/bazel-bin/tensorflow:$LD_LIBRARY_PATH

# Compile LSTM Op in returnn

For example:

    $ ./path-to-your-returnn/tools/compile_native_op.py --native_op NativeLstm2 --output path-to-libraries.txt

The LSTM cell we used is nativelstm2, path-to-libraries.txt contains the paths to the compiled libraries. The libraries are stored by default in /var/tmp/, please move the libraries to somewhere else.  
Alternatively, you can use the option --config to compile all native ops, see the script compile_native_op.py for more detailes.

# Modifications in the original network config file 

1. In the config file, for each LSTM layers, please add 

    "initial_state" : "keep_over_epoch"
    
and change LSTM unit "lstm" to "nativelstm2". We tested that the LSTM unit "lstm" does not work for inference.

2. You have this in your config:

    num_outputs = {"data": {"dim": num_inputs, "sparse": True, "dtype": "int32"}}  # sparse data
    num_outputs["delayed"] = num_outputs["data"]

  Change that to:

    extern_data = {
      "delayed": {"dim": num_inputs, "sparse": True, "dtype": "int32", "available_in_inference": True},
      "data": {"dim": num_inputs, "sparse": True, "dtype": "int32", "available_in_inference": False}}

3. Add the following line to your config file:

    default_input = "delayed"

# Create the Tensorflow graph for inference

    $ ./path-to-your-returnn/tools/compile_tf_graph.py graph_for_inference.config --eval 1 --output_file filename+[".meta", ".metatxt"]
    
.meta graph: the graph for inference.  
.metatxt: contains all the node names of the graph.

**Note**: we will include the .meta graph for inference in the checkpoint.  
Suppose network.040.meta is the original .meta file, network.040.inference.meta is the one we create for inference, replace the original one with the new one by:

    $ mv network.040.inference.meta network.040.meta
    
# Example to use lattice rescorer

## Makefile

Before using the Makefile, please verify the paths to your Tensorflow source code, cuda and cuDNN in Makefile

## Gernerate executable file from source files

In the folder returnn/tools/lattice_rescorer, execute the following command

    $ make
    
If you got the following error while compiling using Makefile:

libopenblasp-r0-39a31c03.2.18.so: cannot open shared object file: No such file or directory  
Please do:

    find -name  libopenblasp-r0-39a31c03.2.18.so

and add the following line in .bashrc:

    export LD_LIBRARY_PATH=/folder-of-the-missing-library:$LD_LIBRARY_PATH
    
## Usage of lattice rescorer tool

lattice_rescorer [OPTION]... [LATTICE]

For command line options information:

    $ lattice_rescorer --help 
    
--ops-Returnn arg  
Text file containing the paths to libraries of the native ops defined in returnn, for more details, please check the part **"Compile LSTM op in returnn"**.

--checkpoint-files arg  
checkpoint of tensorflow model, but we should replace the original .meta graph with the .meta graph for inference created using returnn, please check the part **"Create the TensorFlow graph for inference"**.

--state-vars-list arg  
Text file containing the information needed to assign a value to the state variables in LSTM cell. Please check **example/README.md**.

--tensor-names-list arg  
Text file of tensor names for feeding and fetching. Please chech **example/README.md**.

For the usages of the other options, please check [rwthlm](https://www-i6.informatik.rwth-aachen.de/web/Software/rwthlm.php)
 
Please read an example script example/rescore_lattice.sh. 

## An example script to rescore a lattice

The script rescore_lattice.sh is an example to rescore a lattice.

Before using the script, please modify manually the files libs_list, state_vars_list, tensor_names_list. For the details of these three text files, please read **example/README.md**. And then
    
    $ cd example
    $ ./rescore_lattice.sh