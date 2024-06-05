# Tic-Tac-Joe: A Generative Tic-Tac-Toe Model

## Authors
This project was created by Theodore Tasman and Benjamin Rodgers

## Overview
This project aims to create a generative model for playing tic-tac-toe using a neural network. The model learns to play tic-tac-toe by playing against another iteration of itself.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Future Work](#future-work)
- [License](#license)

## Project Description
The goal of this project is to build a neural network that can play tic-tac-toe by predicting the next optimal move given the current board state. Unlike traditional rule-based approaches, this model uses data-driven methods to learn and generate moves.

## Features
- Train a neural network to play tic-tac-toe
- Evaluate model performance on unseen data
- Deploy the model for human vs AI or AI vs AI gameplay
- Interactive user interface for gameplay

## Installation
To get started with the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/tedtasman/tic-tac-joe.git
    cd tic-tac-joe
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Model Architecture
The model is built using a neural network architecture that takes the current board state as input and outputs the predicted next move. The architecture details are as follows:
- **Input Layer**: Encodes the 3x3 tic-tac-toe board state.
- **Hidden Layers**: Multiple layers with activation functions to learn complex patterns.
- **Output Layer**: Predicts the best next move (position on the board).

## Training
The training process involves putting two iterations of the model against each other in a game of tic-tac-toe. Through an evolutionary process, the model will mutate on every iteration, rewarding wins and punishing loses.

## Evaluation
The model is evaluated on a separate test dataset to measure its accuracy and performance in predicting moves. Evaluation metrics include accuracy and loss.

## Future Work
- **User Interface**: Develop an interactive user interface for playing against the AI.
- **Reinforcement Learning**: Explore reinforcement learning techniques to improve the AI's gameplay.
- **Generalization**: Test the model's performance against different strategies and improve its robustness.

## Changelog
All notable changes to this project will be documented in the [CHANGELOG](CHANGELOG.md).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more details.
