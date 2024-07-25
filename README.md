
# Companion Arbiter PUF (CAR-PUF) Breaker

## Overview

This project aims to demonstrate that a Companion Arbiter PUF (CAR-PUF) can be effectively broken using a linear machine learning model. A CAR-PUF consists of two arbiter PUFs—a working PUF and a reference PUF—and a secret threshold value. Given a challenge, the responses from both PUFs are compared against the threshold to determine the output.

The goal is to derive a linear model that can accurately predict a CAR-PUF's responses based on the provided challenge-response pairs (CRPs). This involves creating a feature vector from the 32-bit challenge such that a linear model can predict the CAR-PUF responses.

## Project Components

1. **Mathematical Derivation**:
   - Detailed mathematical derivation to show how a linear model can break a CAR-PUF.
   - Derivation includes mapping 32-bit challenge vectors to a D-dimensional feature space where a linear model can make accurate predictions.

2. **Python Implementation**:
   - Implementation of the feature extraction and linear model training.
   - Utilizes libraries such as `numpy`, `scikit-learn`, and `scipy` for computations and model training.
   - Includes functions for feature mapping (`my_map`) and model fitting (`my_fit`).

## Key Files

- **`submit.py`**: Contains the implementation of the `my_fit` and `my_map` functions. The `my_fit` function trains the linear model, while the `my_map` function generates the feature vectors from the challenge data.
