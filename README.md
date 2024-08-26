# Model Pruning Example in PyTorch

This repository contains an example of model pruning using PyTorch. It demonstrates how to perform depth and width pruning on a simple neural network model, `SimpleNet`, by creating two pruned versions: `DepthPrunedNet` and `WidthPrunedNet`.

## Table of Contents

- [Introduction](#introduction)
- [Pruning Techniques](#pruning-techniques)
  - [Depth Pruning](#depth-pruning)
  - [Width Pruning](#width-pruning)
- [Real-World Benefits](#real-world-benefits)
- [Code Overview](#code-overview)
- [Installation](#installation)

## Introduction

Pruning is a technique used to reduce the size of a neural network by removing certain parts of it without significantly compromising its performance. This can lead to a more efficient model in terms of computation, memory usage, and inference speed.

In this example, we explore two types of pruning:

- Depth Pruning: Reducing the number of layers.
- Width Pruning: Reducing the number of neurons within a layer.

## Pruning Techniques

### Depth Pruning

Depth pruning involves removing entire layers from the neural network. In the provided code, `DepthPrunedNet` is a pruned version of `SimpleNet` where one of the hidden layers (`layer3`) has been removed. This reduces the model's depth, potentially decreasing the model's computational complexity and memory usage.

### Width Pruning

Width pruning refers to reducing the number of neurons within a layer. In the `WidthPrunedNet` example, the number of neurons in the second layer has been reduced from 128 to 64. This reduces the model's parameter count, potentially leading to a faster and less resource-intensive model.

## Real-World Benefits

Pruning techniques like depth and width pruning offer several benefits in real-world applications:

- **Reduced Model Size:** Pruned models require less storage space, which is crucial for deployment on devices with limited memory.
- **Faster Inference:** Smaller models generally lead to faster inference times, which is essential in real-time applications.
- **Lower Energy Consumption:** Reducing the computational load translates to lower energy usage, which is beneficial for mobile and embedded devices.
- **Maintainable Performance:** With careful pruning, the performance loss can be minimal, allowing for more efficient models without significant accuracy drops.

## Code Overview

The code defines three neural network models:

- `SimpleNet`: The base model with four linear layers.
- `DepthPrunedNet`: A pruned version of `SimpleNet` with one less layer.
- `WidthPrunedNet`: A pruned version of `SimpleNet` with fewer neurons in the second layer.

Each model is initialized and summarized using the `torchsummary` library, and the summaries are logged for inspection.

### Key Files

- `prune.py`: Contains the model definitions, logging setup, and the main function that initializes and summarizes the models.
- `environment.yml`: Defines the conda environment with all necessary dependencies.

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. **Create the Conda Environment and run it**:
   
   ```bash
   conda env create -f environment.yml
   Activate the Environment:
   conda activate myenv
   Run the prune.py
   ```

**Conclusion**
Pruning is a powerful tool for optimizing neural networks, making them more suitable for real-world deployment where resources are limited. This example provides a starting point for implementing and experimenting with pruning techniques in your models.