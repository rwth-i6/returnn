# Before using this file, please verify the paths to your Tensorflow source files, cuda and cuDNN
SRC = main.cc htklatticerescorer.cc vocabulary.cc
OBJ = $(SRC:%.cc=%.o)
DEPENDFILE = .depend
BOOST = /work/speech/tools/boost/boost_1_53_0
INTEL = /work/speech/tools/intel-composer

TENSORFLOW = /u/deng/tensorflow
CUDA=/usr/local/cuda-9.0
CUDNN_v7.0=/u/deng/Downloads/cudnn-9.0-v7.0

# g++ 5.4.0
CXX = g++

# compiler and linker options
CXXFLAGS = -std=c++0x -O2 -m64 -mtune=native -fopenmp -g -D_GLIBCXX_USE_CXX11_ABI=0 \
           -I. -I$(BOOST)/include \
           -I$(INTEL)/ipp/include \
           -I$(INTEL)/mkl/include \
	       -I$(CUDNN_v7)/include \
           -I$(TENSORFLOW)/bazel-tensorflow/external/protobuf_archive/src \
           -I$(TENSORFLOW)/bazel-genfiles -I$(TENSORFLOW) \
           -I$(TENSORFLOW)/bazel-out/k8-py3-opt/genfiles/tensorflow/cc/ops \
           -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/eigen \
           -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/nsync/public \
           -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/protobuf/src

LDFLAGS := -L$(BOOST)/lib -lboost_program_options -lboost_filesystem \
                          -lboost_system -lboost_random -lboost_iostreams -lm \
           -L$(INTEL)/ipp/lib/intel64 -lippcore -lipps \
           -L$(INTEL)/lib/intel64 -liomp5 \
           -L$(INTEL)/mkl/lib/intel64 -lmkl_rt \
           -L$(TENSORFLOW)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow -ltensorflow_framework \
           -lpthread \
           -Wl,-rpath,$(BOOST)/lib \
           -Wl,-rpath,$(INTEL)/ipp/lib/intel64 \
           -Wl,-rpath,$(INTEL)/lib/intel64 \
           -Wl,-rpath,$(INTEL)/mkl/lib/intel64 \
           -Wl,-rpath,$(CUDA)/lib64 \
           -Wl,-rpath,$(CUDNN_v7.0)/lib64

# default rule
all: dep lattice_rescorer

-include $(DEPENDFILE) # "-" means do not stop if file does not exist

lattice_rescorer: $(OBJ)

.PHONY: clean dep

clean:
	rm -f *.o .depend lattice_rescorer

dep: $(SRC)
	$(CXX) $(CXXFLAGS) -MM $(SRC) > $(DEPENDFILE)

# implicit rules (-n show commands, -r no implicit rules)
lattice_rescorer: $(OBJ)
	$(CXX) $(CXXFLAGS) -o lattice_rescorer $(OBJ) $(LDFLAGS)
$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ -c $(@:%.o=%.cc)
