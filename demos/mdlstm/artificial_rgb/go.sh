./create_test_h5.py
mkdir -p models priors log
../../../rnn.py trainconfig
../../../rnn.py forwardconfig
