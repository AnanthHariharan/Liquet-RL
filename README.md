# Liquet-RL

This repository contains a minimal reinforcement learning agent that uses a **Liquid Time-Constant (LTC) neural network** as the recurrent core of an actor-critic policy. The example is designed for the classic `CartPole-v1` environment from `gymnasium` and demonstrates how to implement an LTC cell and train it end-to-end with policy gradients.

## Requirements

```bash
pip install -r requirements.txt
```

## Running the example

```bash
python train.py
```

Training prints the episodic reward every few episodes. The implementation is intentionally compact so it can serve as a starting point for further experimentation.
