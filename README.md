# Machine-Translation
A PyTorch implementation for Machine translation on PHP Corpus dataset based on simple RNN model.

Link for PHP Corpus Data : https://opus.nlpl.eu/PHP.php

---
Whenever running on a different platform, make the following changes in the code:
1. Modify the local path to the dataset given in **data.py**

-------
Things to take care :
1. Fix the dictionary - high
   1. Add words for train/val in/out
2. Add arg parser commands in readme - low
   1. (https://github.com/pytorch/examples/tree/master/word_language_model)
3. Encoder-decoder block - high
4. Update the README template - low
5. Check for validation error - medium
   1. should not be so high 
   2. code has been rectified and divided by the datasamples in validation but still needs to be verified after running for one full epoch
6. Add project requirements from interpretor - Low

-------
For future help in fixes
1. model.py : 
   1. if decoder is fed with "output", the batch size of output of decoder and target is fine, but if decoder is fed with "hidden", batch size of output of decoder and target is not same