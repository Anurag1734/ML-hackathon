# test_env.py
from src.hangman_env import HangmanEnv

# Test the environment
env = HangmanEnv("HELLO", max_attempts=6)

print("Testing Hangman Environment...")
print(f"Word: {env.word}")
print(f"Masked: {env.get_masked_word()}")

# Test some guesses
print("\n--- Testing Guesses ---")
masked, reward, done = env.step('E')
print(f"Guess 'E': {masked}, Reward: {reward}, Done: {done}")

masked, reward, done = env.step('L')
print(f"Guess 'L': {masked}, Reward: {reward}, Done: {done}")

masked, reward, done = env.step('X')
print(f"Guess 'X': {masked}, Reward: {reward}, Done: {done}")

print(f"\nRemaining attempts: {env.remaining_attempts}")
print("Environment working!")

# Complete the word
print("\n--- Completing the Word ---")
for letter in ['H', 'O']:
    masked, reward, done = env.step(letter)
    print(f"Guess '{letter}': {masked}, Reward: {reward}, Done: {done}")

if done:
    print("\n Word completed successfully!")
else:
    print("\n Word not completed yet.")
