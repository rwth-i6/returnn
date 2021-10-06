This is a MDLSTM setup for the IAM handwriting database. Please note, that a GPU is required, since our MDLSTM implementation is GPU only.

If you find the MDLSTM implementation useful, please consider citing the following paper:
Paul Voigtlaender, Patrick Doetsch, Hermann Ney
International Conference on Frontiers in Handwriting Recognition (ICFHR), Shenzhen, China, October 2016, IAPR Best Student Paper Award
https://www-i6.informatik.rwth-aachen.de/publications/download/1014/VoigtlaenderPaulDoetschPatrickNeyHermann--HwritingRecognitionwithLargeMultidimensionalLongShort-TermMemoryRecurrentNeuralNetworks--2016.pdf

This setup includes a small demo with 3 images and can be extended to the full IAM database by obtaining the full dataset (see below)

Instructions for the 3 image demo:
-run ./go.sh

Instructions for the full dataset:
-register at https://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php to get access to the IAM database

then either use this script (creates a new clone of RETURNN): https://gist.github.com/cwig/315d212964542f7f1797d5fdd122891e
or follow the instructions below:
-download and extract https://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz
-copy the .png files from all subdirectories to the IAM_lines folder
-replace the lines.txt by the full file from https://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/lines.txt
-replace the line "python ../../../rnn.py config_demo" in go.sh with "python ../../../rnn.py config_real"
-uncomment the lines "  #convert_IAM_lines_train(base_path_imgs, tag)" and "  #convert_IAM_lines_valid_test(base_path_imgs, tag)" in create_IAM_dataset.py
-run ./go.sh

For forwarding and simple decoding have a look at config_fwd and decode.py
