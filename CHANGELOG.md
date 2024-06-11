# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# Unreleased

## [5.3.0] - 2024/06/10
### Hyperparameters tweaks:
- tieReward = 0
- 2 hidden layers
    - dimensionality 32
- no dropouts

## [5.3.0] - 2024/06/10
### Hyperparameters tweaks:
- tieReward = 0

## [5.2.0] - 2024/06/10
### Hyperparameters tweaks:
- gamma = 0.99

## [5.1.0] - 2024/06/10
### Hyperparameters tweaks:
- alpha = 0.1

## [5.0.3] - 2024/06/10
### Hyperparameters tweaks:
- gamma = 0.93

## [5.0.2] - 2024/06/10
### Hyperparameters tweaks:
- alpha = 0.3

## [5.0.1] - 2024/06/10
### Hyperparameters tweaks:
- tieReward = 0.1

## [5.0.0] - 2024/06/10
### Model:
- Switch to dual model architecture
    - each dualModel object contains a model for play as X and a model for playing as O
    - allows specialization in strategy
    - eliminates any chance of convolution from one model playing both sides
### 5.0.0 Base Hyperparameters:
- alpha = 0.01
- gamma = 0.9
- tieReward = 0.2

## [4.0.1] - 2024/06/08
### Model: 
- Use 1x10 layout to include current move
### Added:
- Better UI when running training
### Fixed:
- Issues with bestValidAction returning nonsense indices

## [3.0.0] - 2024/06/07 **CURRENTLY ABANDONED**
### Model 
- attempting backprop

# Released

## [2.4.2] - 2024/06/07 
### Fixed
- inverted rewards
- various bugs in training code
### Issues
- Model seemingly learns nothing

## [2.3.0] - 2024/06/07
### Model:
- One dropout
-Two hidden
### Hyperparameters:
- epsilon = 1-0.9**(iterations/i) (exponential decay; more random in the beginning)
- discountFactor = 0.9
- learningRate = 0.0024

## [2.2.0] - 2024/06/07
### Hyperparameters:
- epsilon = 0.1
- discountFactor = 0.9
- learningRate = 0.0024

## [2.1.0] - 2024/06/07
### Hyperparameters:
- epsilon = 0.1
- discountFactor = 0.9
- learningRate = 0.0012

## [2.0.0] - 2024/06/07
### Added
- Q Learning
- Fancy play UI
### Hyperparameters:
- epsilon = 0.1
- discountFactor = 0.9
- learningRate = 0.0001

## [1.2.0] - 2024-06-06
### Added
- Improved usability overall
- Load and save models
- AI vs AI mode
### Fixed
- Stuck in user vs AI bug
### Issues
- Model is learning but not improving. Seems to be overfitting.

## [1.0.1] - 2024-06-05
### Fixed
- Infinite loop when playing against Joe
### Issues
- Model is not learning (tested at 15000 iterations, no success)

## [1.0.0] - 2024-06-05
### Added
- Neural network to play tic-tac-toe
- Training loop to train AI
### Issues
- Playing against trained AI broken - infinite loop

## [0.3.0] - 2024-06-05
### Added
- inOut class to interactively play game
    - features runFree method to automatically call required functions

## [0.2.0] - 2024-06-04
### Added
- README.md with project overview and setup instructions.
- MIT License.
- requirements.txt to hold dependencies.
- CHANGELOG.md to log progress.

## [0.1.0] - 2024-06-03
### Added
- board.py for encoding the 3x3 tic-tac-toe board which allows:
    - Playing moves at a given position.
    - Checking for win conditions.

