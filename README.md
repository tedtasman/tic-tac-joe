# Tic-Tac-Joe: A Deep-Learning Tic-Tac-Toe Model

## Authors
This project was created by Theodore Tasman and Benjamin Rodgers

## Overview
This project aims to create a deep-learning model for playing tic-tac-toe using a neural network. The model learns to play tic-tac-toe through Q-Learning.

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
The training process involves playing an AI agent against semi-random play for many epochs. For every move, the agent will either exploit its training; using the neural network to predict the best move, or explore; generate a random move. Utilizing Q-Learning techniques, the model refits after every epoch rewarding wins and punishing loses. 

## Evaluation
The model is evaluated on its performance versus the training program, as well as im human interaction. By graphing the win percentage over time, overfitting can be identified and the ideal number of epochs deduced.

## Future Work
- **Live Demo**: Host the program live on [ttasman.com](https://ttasman.com/) 

## Changelog
All notable changes to this project will be documented in the [CHANGELOG](CHANGELOG.md).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more details.
