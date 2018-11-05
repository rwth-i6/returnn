#!/bin/bash
# Please check ../README.md for more details about the command line options

export OMP_NUM_THREADS=2
lattice_rescorer="../lattice_rescorer"
Dir="."
Vocab=<path-to-your-vocabulary> # Please verify the vocabulary before using the script
LibsReturnn="${Dir}/libs_list"
CheckPoint="${Dir}/network.040/network.040"

$lattice_rescorer \
    --vocab ${Vocab} \
    --lambda 0.715557 \
    --pruning-threshold 50 \
    --dp-order 9 \
    --look-ahead-semiring none \
    --lm-scale 13.4 \
    --output expanded-lattice \
    --set-sb-last 1 \
    --ops-Returnn ${LibsReturnn} \
    --checkpoint-files ${CheckPoint} \
    --state-vars-list ${Dir}/state_vars_list \
    --tensor-names-list ${Dir}/tensor_names_list \
    ${Dir}/QRBC_ENG_GB_20110119_120000_BBC_WH_POD_0000313109_0000325199.lat.gz

