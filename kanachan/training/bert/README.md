# `kanachan.training.bert` Python submodule

## Note

The use of the term **BERT** here is a deliberate abuse. The term BERT actually refers to a combination of a model with transformer encoder layers and a learning method for that model. However, this project uses the term BERT to refer only to the model. The model should actually be called something like transformer encoder layers, but that would be too long, so this project calls it BERT.

## Submodules

### [`kanachan.training.bert.phase1`](phase1) Python submodule

A training program for BERT with an objective function that imitates human choices in training data.

### [`kanachan.training.bert.phase2`](phase2) Python submodule

A training program for BERT with an objective function that maximizes round deltas in training data.
