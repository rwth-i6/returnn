
# Create/fix files
Please check the corresponding samples.

1. libs_list: It contains the path to the compiled library of LSTM op, for more details, please check README.md up one directory **"Compile LSTM Op in returnn"**.

2. state_vars_list: It contains the information needed for the assignment of the state variables in LSTM cell. The format of each row is:

    - (name of state variable) (name of assignment node of the state variable) (name of the other input node of the assignment) (size of the state variable)

    - The above information now should be found mannually in the .metatxt file. For more details, please check README.md up one directory **"Create the Tensorflow graph for inference"**. 

3. tensor_names_list: It contains the names of tensors needed for feeding and fetching

# Needed additional files

network.040:(model, TF checkpoint, computation graph)  
vocab.txt  
network.040.data*  
network.040.index  
network.040.meta  
**Note**: .meta graph is **NOT** the original one in TF checkpoint, the original one is replaced by the .meta graph created using returnn. For more details, check README.md up one directory **"Create the Tensorflow graph for inference"**.

.:
example lattice (QRBC_ENG_GB_20110119_120000_BBC_WH_POD_0000313109_0000325199.lat.gz)
