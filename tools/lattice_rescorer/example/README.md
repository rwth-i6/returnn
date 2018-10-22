
# Create/fix files

libs_list (It contains the path to the compiled library of lstm op)
state_vars_list (The first colume is the name of state var, the second colume is its dimensionality)
tensor_names_list (It contains the names of tensors needed for forwarding)


# Needed additional files

network.040:  # (model, TF checkpoint, computation graph)
vocab.txt
network.040.data*
network.040.index
network.040.meta (Note:This meta graph is NOT the original one in TF checkpoint, the original one is replaced by the meta graph created using Returnn, refer to lattice_rescorer/README.md)

.:
example lattice (QRBC_ENG_GB_20110119_120000_BBC_WH_POD_0000313109_0000325199.lat.gz)
