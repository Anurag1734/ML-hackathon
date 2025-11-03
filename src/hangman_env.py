# Fixed HangmanEnv with 'done' and 'is_won()' attributes
class HangmanEnv:
    def __init__(self, word, max_attempts=6):
        self.word = word.upper()
        self.max_attempts = max_attempts
        self.remaining_attempts = max_attempts
        self.guessed = set()
        self.done = False  # ADD THIS
    
    def get_masked_word(self):
        return ''.join(c if c in self.guessed else '_' for c in self.word)
    
    def step(self, letter):
        letter = letter.upper()
        
        if letter in self.guessed:
            reward = -0.5  # repeated guess
        elif letter in self.word:
            self.guessed.add(letter)
            reward = 1
        else:
            self.guessed.add(letter)
            self.remaining_attempts -= 1
            reward = -1
        
        # Update done status
        self.done = (self.remaining_attempts == 0) or ('_' not in self.get_masked_word())
        
        return self.get_masked_word(), reward, self.done
    
    def reset(self, word):
        """Reset environment for new game."""
        self.word = word.upper()
        self.remaining_attempts = self.max_attempts
        self.guessed = set()
        self.done = False  # ADD THIS
    
    def is_won(self):
        """Check if game is won"""
        return '_' not in self.get_masked_word()
    
    def is_lost(self):
        """Check if game is lost"""
        return self.remaining_attempts <= 0

print("✓ Fixed HangmanEnv loaded")
