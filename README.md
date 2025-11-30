# QER-Demo: Core Implementation of Quantum-Inspired Sequence Experience Replay

This repository contains the official core implementation of the **QER (Quantum-Inspired Sequence Experience Replay)** algorithm, as proposed in our paper:

> **QER-LPD3QN: A Quantum-Inspired Sequence-Aware Deep Reinforcement Learning Algorithm for Path Planning**

## 📂 Repository Contents

This repository is designed to facilitate the reproduction of the proposed **QER mechanism** and verify the "Prepare" and "Depreciate" operators described in the manuscript.

* **`qer_buffer.py`**: The complete, standalone implementation of the `QuantumSequencePrioritizedReplayBuffer` class. It includes:
    * **Quantum State Representation**: Encoding priorities as amplitudes on the Bloch sphere.
    * **Prepare Operation**: Rotation logic driven by TD-error (Eq. 31-35 in the paper).
    * **Depreciate Operation**: Rotation logic driven by replay counts (Eq. 36-37 in the paper).
    * **Sequence Sampling**: Sliding window sampling based on quantum probabilities.
* **`demo_toy.py`**: A verification script that runs the QER algorithm with mock data. It demonstrates the state update cycle without requiring the full simulation environment.
* **`output2.gif` / `output3.gif`**: Visual demonstrations of the agent navigating in highly dynamic environments using the QER-LPD3QN planner.

## 🚀 Quick Start

You can verify the algorithmic logic immediately using the provided demo script. No complex dependencies (like ROS or Gazebo) are required.

### 1. Requirements
* Python 3.x
* NumPy

### 2. Run the Verification Demo
```bash
python demo_toy.py
