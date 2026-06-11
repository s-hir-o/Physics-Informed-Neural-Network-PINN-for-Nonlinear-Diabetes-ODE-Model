# Physics-Informed-Neural-Network-PINN-for-Nonlinear-Diabetes-ODE-Model


## Overview

This repository presents a Physics-Informed Neural Network (PINN) implementation for solving a nonlinear, coupled system of ordinary differential equations that model diabetes progression across multiple population compartments.

The approach integrates domain-specific physical laws directly into the training process, allowing the neural network to learn a continuous-time solution that satisfies the governing dynamics without relying on traditional numerical solvers.

---

## Problem Description

The model consists of five interacting compartments:

* S(t): Susceptible population
* D(t): Diabetic population
* TN(t): Non-pharmacological treatment group
* TP(t): Pharmacological treatment group
* R(t): Controlled or recovered population

The system is characterized by:

* First-order differential equations
* Nonlinear interactions
* Coupled dynamics between compartments
* Stiff behavior due to multiple time scales

---

## Methodology

### Physics-Informed Neural Network

A feedforward neural network is trained to approximate the solution:

t → [S(t), D(t), TN(t), TP(t), R(t)]

The network is optimized by minimizing a composite loss function that enforces both data-independent physical constraints and system dynamics.

---

## Loss Function Design

The training objective consists of the following components:

### 1. Physics Loss

Enforces all governing differential equations, including:

* Compartment dynamics
* Total population evolution

### 2. Initial Condition Loss

Ensures that the predicted solution matches physically consistent initial population values.

### 3. Population Constraint

Imposes the boundedness condition:

N(t) = S + D + TN + TP + R ≤ Λ / μ

This guarantees biologically meaningful solutions throughout training.

---

## Key Features

* Solves a five-dimensional nonlinear ODE system
* Enforces physical constraints directly within the loss function
* Uses automatic differentiation for computing derivatives
* Handles stiff dynamics through constraint-driven learning
* Produces continuous-time predictions without numerical integration

---

## Implementation Details

* Framework: PyTorch
* Network architecture: Fully connected feedforward network
* Activation: Tanh (hidden layers), Softplus (output layer for positivity)
* Training: Gradient-based optimization using Adam
* Time normalization applied for numerical stability

---

## Results

The trained model exhibits the expected qualitative behavior:

* Decreasing susceptible population
* Growth and stabilization of diabetic population
* Evolution of treatment compartments
* Bounded total population over time

A representative output plot is provided in `results.png`.

---

## Project Structure

```
.
├── pinn_model.py      # Neural network architecture
├── results.png        # Output plot
├── requirements.txt   # Dependencies
└── README.md
```

---

## Installation

Install dependencies using:

```
pip install -r requirements.txt
```

---

## Usage

Train the model:

```
python train.py
```

Run inference and generate plots:

```
python predict.py
```

---

## Skills Demonstrated

* Physics-informed machine learning
* Nonlinear dynamical systems modeling
* Scientific computing with PyTorch
* Constraint-based optimization
* Handling stiff systems in neural networks

---

## References

Diabetes compartmental model based on a published research study.


---

## Notes

This implementation focuses on solving the forward ODE system using PINNs.
Parameter estimation and optimal control formulations are not included.

---

## Author

GitHub: https://github.com/s-hir-o

