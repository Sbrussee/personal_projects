# Personal coding projects
Code repository for personal coding projects, mostly in the field of bioinformatics/AI.

Here follows a short description of very project so far:
## Transcription factor binding predictor using CNNs
In this project, 2 CNN models were built capable of accurately predicting transcription factor binding based on an input DNA-sequence. One of the models used raw DNA sequences which were fed through a convolution layer multiple times. The other model used an embedding (similar to word2vec) for encoding the DNA sequences, after which roughly the same architecture as in the first model was used. Embedding the DNA sequences allowed for faster convergence of the model accuracy.
