How to use the lattice rescorer tool.



# Install Tensorflow from source

(refering to https://www.tensorflow.org/install/install_sources )

## Clone the Tensorflow repository
    
To clone the latest TensorFlow repository, issue the following command:

    $ git clone https://github.com/tensorflow/tensorflow 

The preceding git clone command creates a subdirectory named "tensorflow".
    
## Prepare environment for Linux
    
  Before building TensorFlow on Linux, install the following build tools on your system:
  *bazel
  *TensorFlow Python dependencies
  *optionally, NVIDIA packages to support TensorFlow for GPU.
    
### Install bazel without root access

Note: expected at least bazel 0.5.4, otherwise build pip package will fail.
       (https://docs.bazel.build/versions/master/install-ubuntu.html#installing-using-binary-installer)
download bazel-0.11.1-installer-linux-x86_64.sh
$ chmod +x bazel-0.11.1-installer-linux-x86_64.sh
$ ./bazel-0.11.1-installer-linux-x86_64.sh --user
the Bazel executable is installed in your $HOME/bin directory.
	add export PATH="$PATH:$HOME/bin" in ~/.bashrc

### Install Tensorflow Python dependencies

To install TensorFlow, you must install the following packages:numpy, dev, pip, wheel
(The required packages above are already installed on computers at i6)

### Install TensorFlow for GPU prerequisites

The following NVIDIA hardware must be installed on your system:
*GPU card with CUDA Compute Capability 3.0 or higher. See NVIDIA documentation for a list of 
supported GPU cards.
The following NVIDIA software must be installed on your system:
*CUDA Toolkit (>= 7.0). We recommend version 9.0. For details, see NVIDIA's documentation. 
Ensure that you append the relevant CUDA pathnames to the LD_LIBRARY_PATH environment 
variable as described in the NVIDIA documentation.
*GPU drivers supporting your version of the CUDA Toolkit.
*cuDNN SDK (>= v3). We recommend version 7.0. For details, see NVIDIA's documentation.

Note: tensorflow c++ library requires libcudnn.so.7, but the installed cudnn is cudnn-8.0, so we have 
to install cuDNN 7 locally:
download the cudnn 7.0.5 for cuda 9.0,https://developer.nvidia.com/rdp/cudnn-archive
before that you have to register. 

    $ tar -xzvf cudnn-9.0-linux-x64-v7.tgz
    $ cp cuda/lib64/libcudnn.so.7 /u/username/.local/lib
    $ cp cuda/include/cudnn.h  /u/username/.local/include

Then in the file ~/.bashrc add one line:
export LD_LIBRARY_PATH=/u/username/.local/lib:LD_LIBRARY_PATH

*CUPTI ships with the CUDA Toolkit, but you also need to append its path to the 
LD_LIBRARY_PATH environment variable in the file .bashrc by adding the following line:   
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 


## Configure the installation

Identify the pathname of all relevant TensorFlow dependencies and specify other build configuration 
    options such as compiler flags. 
    $ cd tensorflow  # cd to the top-level directory created
    $ ./configure   
    Note: two configuration questions you should notice
1. Please specify the CUDA SDK version you want to use: (Please enter  9.0)
2. Please specify the cuDNN version you want to use.(Please enter 7.0)
   1.4 Build the pip package
To build a pip package for TensorFlow with GPU support, invoke the following command:
            $ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
 
            Note: gcc 5 or later: the binary pip packages available on the TensorFlow website are built with   
gcc 4, which uses the older ABI. If you want to make your build compatible with the older ABI, you need to add --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" to your bazel build command.
Running the script "build_pip_package" as follows will build a .whl file within the /tmp/tensorflow_pkg directory:
Note: Please use another directory instead of /tmp/ to make .whl file reusable
            $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

            
## Installl the pip package

The filename of the .whl file depends on your platform. For example,

    $ pip install --user /tmp/tensorflow_pkg/tensorflow-1.8.0-py2-none-any.whl

Note: the option --user must be added for the users without root access.


# Compiling tensorflow c++ library from source

Note: before compiling the libraries using bazel, please modify the outputRoot directory of bazel, otherwise the default path is /u/username/.cache/bazel, which would lead to some problems that the soft links to this directory are invalid.
$ export TEST_TMPDIR=/u/username/bazel_outputRoot for example
build the c++ library on GPU:(on CPU, some libraries can not be found)
$ bazel build -c opt --config=cuda --local_resources=6144,4,1.0 --jobs=4 //tensorflow:libtensorflow_cc.so
build the c library:
$ bazel build -c opt --config=cuda --local_resources=6144,4,1.0 --jobs=4 //tensorflow:libtensorflow.so
directly download the c library by invoking the following shell commands:
(refering to https://www.tensorflow.org/install/install_c)
$ TF_TYPE="gpu"
$ OS="linux"
$ TARGET_DIRECTORY="/u/username/tensorflow/bazel-bin/tensorflow"
$ curl -L \"https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}
-${OS}-x86_64-1.8.0.tar.gz" | tar -C $TARGET_DIRECTORY -xz

In the file .bashrc add the line below:
export LD_LIBRARY_PATH=/u/username/tensorflow/bazel-bin/tensorflow: LD_LIBRARY_PATH


# Compile LSTM Op in RETURNN

In RETURNN project, invoke the following shell commands:

    $ ./path-to-your-RETURNN/tools/compile_native_op.py --native_op LstmGenericBase --output xyz

xyz contains the paths to LstmGenericBase.so, copy it somewhere save(by default it is stored in /var/tmp/)
The compiled libraries are _lstm_ops.so, LstmGenericBase.so and GradOfLstmGenericBase.so


# Create the Tensorflow graph for forwarding

In RETURNN

    $ ./path-to-your-RETURNN/tools/compile_tf_graph.py <filename to config-file> --eval 1 --output_file <filename to output pb or pbtxt file>


