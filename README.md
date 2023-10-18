# Czech pos-tagger using transformers
This project is an implementation of transformer from scratch in PyTorch (as in `Attention is all you need`). The task is to map from words to part of speech tags in Czech. The purpose was to have fun and to practise.

The project uses czech_pdt dataset of about 50k sentences. These are split into train/dev/test but I only use train/dev. Each item in dataset is a sentence and contains:
  1) words in their original form - `forms`
  2) words in their dictionary form - `lemmas`
  3) part of speech tags - `tags`

Everything is written for PyTorch. The training is done via teacher forcing and inference via auto-regressive decoding with a greedy next token search. I also added wandb support to monitor training instead of tensorboard because I believe it is superior. But I also added an input layer dropout for experimental reasons. It achieves about 91% acc on dev set and takes about 4 minutes/epoch to train on T4 Google Collab. I am sure further improvements can be achieved by optimizing hyperparams.

How to run:

1) clone repo
2) download data, URL is in Morpho.py
3) run train_transformer.py
