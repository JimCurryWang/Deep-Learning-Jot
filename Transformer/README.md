# About 

Note for implementing a sequence-to-sequence (seq2seq) models with PyTorch and TorchText.


# Environment 

```cmd
python=3.9
torch=1.8.1
torchtext=0.9.1
spacy=3.0.6
```

# Note

* 1 - [Sequence to Sequence Learning with Neural Networks](https://github.com/JimCurryWang/Deep-Learning-Jot/blob/seq2seq-1/Transformer/1_Sequence_to_Sequence_Learning_with_Neural_Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FGqD8Kv-rD3pbDpOTVT9I-WIUeLqAYum?usp=sharing)

    - In this series we'll be building a machine learning model to go from once sequence to another, using PyTorch and torchtext.
    - This will be done on German to English translations with [Multi30k dataset\*](https://github.com/multi30k/dataset).
    - spaCy to assist in the tokenization of the data. 
    - The model itself will be based off an implementation of [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), which uses multi-layer LSTMs.
    - \*Multi30k is a dataset with ~30,000 parallel English, German and French sentences, each with ~12 words per sentence. 


# Reference 

- https://arxiv.org/pdf/1409.3215.pdf
- https://github.com/bentrevett
- https://github.com/aladdinpersson