# Machine-Translation
A PyTorch implementation for Machine translation on PHP Corpus dataset based on simple RNN model.

**Link for PHP Corpus Data :** https://opus.nlpl.eu/PHP.php

---
Whenever running on a different platform, make the following changes in the code:
1. Modify the local path to the dataset given in **data.py**

---
**Package Requirements:**

* Basic : torch, numpy
* File/Data handling : re (or re2), unicodedata (or unicodedata2), sklearn, os
* Misc : time, math, matplotlib, tqdm, argparse

-------
**Things to take care :**
1. Add arg parser commands in readme - low
   1. (https://github.com/pytorch/examples/tree/master/word_language_model)
3. Update the README template - low 
4. print test loop error 
5. Add epochs/batches
6. Add val loop
-------
**For future help in fixes :**
1. model.py : 
   1. if decoder is fed with "output", the batch size of output of decoder and target is fine, but if decoder is fed with "hidden", batch size of output of decoder and target is not same


---
**Updates made recently:**
