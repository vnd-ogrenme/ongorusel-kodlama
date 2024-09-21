### Representation Learning with Contrastive Predictive Coding

This repository contains a Keras implementation of the algorithm presented in the paper [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748).

### Usage

- Execute ```python train_model.py``` to train the CPC model.
- Execute ```python benchmark_model.py``` to train the MLP on top of the CPC encoder.

IMPORTANT: If you want to train the HAR(human activity recognition) model, change the part in the train_model.py accordingly.

### Requisites

- [Anaconda Python 3.5.3](https://www.continuum.io/downloads)
- [Keras 2.0.6](https://keras.io/)
- [Tensorflow 1.4.0](https://www.tensorflow.org/)
- GPU for fast training.

### References

- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
