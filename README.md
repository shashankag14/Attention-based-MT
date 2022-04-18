# Attention based Machine-Translation
(In progress) A PyTorch implementation for Machine translation on PHP Corpus dataset based on simple RNN model.

**Link for PHP Corpus Data :** https://opus.nlpl.eu/PHP.php

---
Whenever running the code, make the following changes in the code:
1. Modify the local path to the dataset given in **data.py**

-------
**To Dos :**
1. Add arg parser commands in readme 
2. Encoder-decoder block  
3. Validation error is low, but train error is high. -  HIGH
4. Save metrices in csv for future usages 
5. Refine plots
-------
**For future help in fixes :**
1. model.py : 
   1. if decoder is fed with "output", the batch size of output of decoder and target is fine, but if decoder is fed with "hidden", batch size of output of decoder and target is not same
