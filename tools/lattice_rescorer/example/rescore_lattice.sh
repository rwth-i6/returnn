#!/bin/bash

export OMP_NUM_THREADS=2
lattice_rescorer="../lattice_rescorer"
Dir="."
Vocab="${Dir}/network.040/vocab.txt"
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

