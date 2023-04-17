# `kanachan.training` Python submodule

## Submodules

### [`kanachan.training.bert`](bert) Python submodule

Training programs for [BERT](https://arxiv.org/abs/1810.04805)[^BERT].

### [`kanachan.training.iql`](iql) Python submodule

Training programs for [implicit Q-learning (IQL)](https://arxiv.org/abs/2110.06169).

### [`kanachan.training.ilql`](ilql) Python submodule

Training programs for [implicit language Q-learning (ILQL)](https://arxiv.org/abs/2206.11871)[^ILQL].

[^BERT]: The use of the term **BERT** here is a deliberate abuse. The term BERT actually refers to a combination of a model with transformer encoder layers and a learning method for that model. However, this project uses the term BERT to refer only to the model. The model should actually be called something like transformer encoder layers, but that would be too long, so this project calls it BERT.

[^ILQL]: The use of the term **ILQL** here is a deliberate abuse. The term ILQL actually refers to a combination of a variant of IQL in which the Q and V models share parameters and a learning method for that model. However, this project uses the term ILQL to refer only to the model. The model should actually be called something like "a variant of IQL with parameter sharing between the Q and V models", but that would be too long, so this project calls it ILQL.
