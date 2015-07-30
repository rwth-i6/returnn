#!/bin/bash

set -e

cd $(dirname $0)
../rnn.py demo-task12ax.config $*

echo "finished. deleting models."
rm /tmp/crnn-task12ax-network.00*
