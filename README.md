# TRRL Framework

This repository contains the implementation of the **Trust Region Reinforcement Learning (TRRL)** framework, along with baseline implementations for comparison, including AIRL, GAIL, BC, Daggle, and SQIL. The environment used for testing is `Pong`, a standard benchmark in reinforcement learning research.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Installation](#installation)
4. [How to Run](#how-to-run)
5. [Baselines](#baselines)
6. [Project Structure](#project-structure)
7. [Acknowledgments](#acknowledgments)

---

## Introduction

**TRRL** is a novel reinforcement learning algorithm that focuses on efficiently solving control tasks using trust region methods to improve stability and performance. This repository allows researchers and practitioners to evaluate TRRL against several state-of-the-art baselines, including:

- **AIRL (Adversarial Inverse Reinforcement Learning)**
- **GAIL (Generative Adversarial Imitation Learning)**
- **BC (Behavior Cloning)**
- **Daggle**
- **SQIL (Soft Q Imitation Learning)**

All experiments are conducted in the **Pong** environment from the `Gymnasium` library.

---

## Environment Setup

1. **Primary Environment**: `Pong` (from `gymnasium`'s Atari environments).
2. **Dependencies**: The code requires Python 3.8 or later and various Python libraries (see `requirements.txt`).

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary environment installed:
   ```bash
   pip install "gymnasium[atari,accept-rom-license]"
   ```

---

## How to Run

### Running the TRRL Method

Run the TRRL implementation:
```bash
python trrl.py
```

### Running Baselines

To run specific baseline algorithms, use the corresponding script:
- **AIRL**: `python AIRL.py`
- **GAIL**: `python GAIL.py`
- **Behavior Cloning (BC)**: `python BC.py`
- **Daggle**: `python Daggle.py`
- **SQIL**: `python SQIL.py`

All scripts are configured to use the `Pong` environment by default.

---

## Baselines

The repository includes several baseline algorithms for comparison:
- **AIRL**: Adversarial approach for inverse reinforcement learning.
- **GAIL**: Generative adversarial approach for imitation learning.
- **BC**: Simple supervised learning approach to mimic expert actions.
- **Daggle**: A state-of-the-art algorithm tailored for imitation learning tasks.
- **SQIL**: Reinforcement learning approach that incorporates expert demonstrations into soft Q-learning.

Each baseline is implemented in a separate script for modularity and clarity.

---

## Project Structure

```plaintext
TRRL/
├── AIRL.py               # AIRL baseline implementation
├── BC.py                 # Behavior Cloning implementation
├── Daggle.py             # Daggle baseline implementation
├── GAIL.py               # GAIL baseline implementation
├── SQIL.py               # SQIL baseline implementation
├── trrl.py               # Main TRRL algorithm implementation
├── arguments.py          # Argument parsing for all scripts
├── reward_function.py    # Reward function utilities
├── rollouts.py           # Utilities for rollout collection
├── model/                # Directory for saved models
├── logs/                 # Directory for logging outputs
└── requirements.txt      # Required dependencies
```

---

## Acknowledgments

This repository builds upon the following libraries:
- [Gymnasium](https://farama.org/Gymnasium/) for environments.
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for reinforcement learning algorithms.
- [Imitation](https://imitation.readthedocs.io/) for inverse reinforcement learning implementations.

If you use this repository in your work, please consider citing the relevant libraries and frameworks.

---

### Additional Notes

- Ensure that your machine has sufficient memory when running Atari environments, as they can be resource-intensive.
- For issues or questions, please contact the repository owner.

--- 
