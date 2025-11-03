# ML-hackathon

## Overview
An intelligent Hangman game-solving agent combining Hidden Markov Models (HMM) for probabilistic letter prediction and Reinforcement Learning (RL) for optimal decision-making.

## Problem Statement
Build a hybrid ML system that:
1. Uses HMM to model letter probability distributions
2. Employs RL agent to make strategic letter guesses
3. Minimizes wrong guesses while maximizing success rate

## Project Components

### Part 1: Hidden Markov Model
- **Purpose**: Estimate probability distribution of letters given current game state
- **Training Data**: 50,000-word corpus
- **Output**: Probability vector over alphabet for each blank position

### Part 2: Reinforcement Learning Agent
- **Environment**: Custom Hangman game environment
- **State**: Masked word + guessed letters + lives remaining + HMM probabilities
- **Actions**: Guess any unguessed letter
- **Reward**: Designed to maximize success and minimize wrong/repeated guesses

## Evaluation Metrics
Final Score = (Success Rate × 2000) - (Total Wrong Guesses × 5) - (Total Repeated Guesses × 2)


Evaluated on 2000 test games with 6 lives per game.