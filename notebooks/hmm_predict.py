import numpy as np
from collections import defaultdict, Counter
import pickle
import string
import re

class ImprovedHangmanHMM:
    """
    Improved Hidden Markov Model for Hangman with multiple strategies
    """
    
    def __init__(self, max_word_length=30):
        self.max_word_length = max_word_length
        self.alphabet = list(string.ascii_lowercase)
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.alphabet)}
        
        # Store all words grouped by length for pattern matching
        self.words_by_length = defaultdict(list)
        
        # HMM parameters by word length
        self.models_by_length = {}
        
        # Global statistics
        self.global_letter_freq = None
        self.common_first_letters = None
        self.common_last_letters = None
        self.vowels = set('aeiou')
        self.common_consonants = set('tnrslhdcmpfgbywkvxjqz')
        
    def train(self, corpus_file):
        """Train HMM on corpus"""
        print("Loading corpus...")
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        print(f"Training on {len(words)} words...")
        
        # Store words by length for pattern matching
        for word in words:
            if len(word) <= self.max_word_length and word.isalpha():
                self.words_by_length[len(word)].append(word)
        
        # Calculate global letter frequency (sorted by frequency)
        all_letters = ''.join(words)
        letter_counts = Counter(all_letters)
        total = sum(letter_counts.values())
        self.global_letter_freq = {letter: letter_counts.get(letter, 0) / total 
                                   for letter in self.alphabet}
        
        # Common first and last letters
        first_letters = Counter([w[0] for w in words if len(w) > 0])
        last_letters = Counter([w[-1] for w in words if len(w) > 0])
        self.common_first_letters = {k: v/len(words) for k, v in first_letters.items()}
        self.common_last_letters = {k: v/len(words) for k, v in last_letters.items()}
        
        # Train models for each word length
        for length, word_list in self.words_by_length.items():
            print(f"Training model for length {length} with {len(word_list)} words")
            self.models_by_length[length] = self._train_length_model(word_list, length)
        
        print("Training complete!")
        print(f"Most common letters: {sorted(self.global_letter_freq.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    def _train_length_model(self, words, length):
        """Train HMM for specific word length"""
        model = {
            'length': length,
            'emission_probs': np.zeros((length, 26)),
            'letter_freq': np.zeros(26),
            'position_letter_counts': np.zeros((length, 26)),
            'letter_bigrams': {},
            'position_patterns': defaultdict(Counter)
        }
        
        # Count emissions
        for word in words:
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    letter_idx = self.letter_to_idx[letter]
                    model['position_letter_counts'][pos][letter_idx] += 1
                    model['letter_freq'][letter_idx] += 1
            
            # Letter bigrams (what follows what)
            for i in range(len(word) - 1):
                if word[i] not in model['letter_bigrams']:
                    model['letter_bigrams'][word[i]] = Counter()
                model['letter_bigrams'][word[i]][word[i+1]] += 1
            
            # Position patterns (what's common before/after positions)
            for pos in range(len(word)):
                if pos > 0:
                    model['position_patterns'][(pos, 'before')][word[pos-1]] += 1
                if pos < len(word) - 1:
                    model['position_patterns'][(pos, 'after')][word[pos+1]] += 1
        
        # Calculate emission probabilities with smoothing
        alpha = 0.001
        for pos in range(length):
            total = model['position_letter_counts'][pos].sum() + alpha * 26
            for letter_idx in range(26):
                model['emission_probs'][pos][letter_idx] = \
                    (model['position_letter_counts'][pos][letter_idx] + alpha) / total
        
        # Normalize letter frequency
        total_letters = model['letter_freq'].sum()
        if total_letters > 0:
            model['letter_freq'] = model['letter_freq'] / total_letters
        
        # Normalize bigrams
        for letter, counter in model['letter_bigrams'].items():
            total = sum(counter.values())
            model['letter_bigrams'][letter] = {k: v/total for k, v in counter.items()}
        
        return model
    
    def _match_pattern(self, masked_word, guessed_letters, word_list):
        """Find words matching the current pattern"""
        pattern = masked_word.replace('_', '.')
        regex = re.compile('^' + pattern + '$')
        
        # Get all wrong letters (guessed but not in masked word)
        wrong_letters = set()
        for letter in guessed_letters:
            if letter not in masked_word:
                wrong_letters.add(letter)
        
        matching_words = []
        for word in word_list:
            if regex.match(word):
                # Check that word doesn't contain any wrong letters
                if not any(letter in word for letter in wrong_letters):
                    matching_words.append(word)
        
        return matching_words
    
    def predict_letter_probabilities(self, masked_word, guessed_letters):
        """
        Predict probability distribution using multiple strategies
        """
        word_length = len(masked_word)
        letter_scores = defaultdict(float)
        
        # Count revealed letters and blanks
        num_blanks = masked_word.count('_')
        num_revealed = word_length - num_blanks
        
        # Get matching words from corpus
        if word_length in self.words_by_length:
            matching_words = self._match_pattern(
                masked_word, 
                guessed_letters, 
                self.words_by_length[word_length]
            )
        else:
            matching_words = []
        
        # Strategy weights (dynamic based on game state)
        pattern_weight = 0.0
        hmm_weight = 0.0
        global_weight = 0.0
        context_weight = 0.0
        
        # Early game (mostly blanks): rely on frequency
        if num_revealed <= 2:
            global_weight = 0.6
            hmm_weight = 0.3
            context_weight = 0.1
        # Mid game (some revealed): balance all strategies
        elif num_revealed <= word_length * 0.5:
            if matching_words and len(matching_words) < 100:
                pattern_weight = 0.5
                hmm_weight = 0.2
                global_weight = 0.2
                context_weight = 0.1
            else:
                global_weight = 0.4
                hmm_weight = 0.4
                context_weight = 0.2
        # Late game (mostly revealed): heavily favor pattern matching
        else:
            if matching_words and len(matching_words) < 50:
                pattern_weight = 0.8
                hmm_weight = 0.1
                global_weight = 0.05
                context_weight = 0.05
            else:
                hmm_weight = 0.5
                global_weight = 0.3
                context_weight = 0.2
        
        # Strategy 1: Pattern matching
        if matching_words and pattern_weight > 0:
            pattern_scores = defaultdict(int)
            for word in matching_words:
                for pos, char in enumerate(masked_word):
                    if char == '_':
                        letter = word[pos]
                        if letter not in guessed_letters:
                            pattern_scores[letter] += 1
            
            if pattern_scores:
                total = sum(pattern_scores.values())
                for letter, count in pattern_scores.items():
                    letter_scores[letter] += (count / total) * pattern_weight
        
        # Strategy 2: Position-based HMM emissions
        if word_length in self.models_by_length and hmm_weight > 0:
            model = self.models_by_length[word_length]
            hmm_scores = np.zeros(26)
            
            for pos, char in enumerate(masked_word):
                if char == '_':
                    hmm_scores += model['emission_probs'][pos]
            
            if hmm_scores.sum() > 0:
                hmm_scores = hmm_scores / hmm_scores.sum()
                for letter in self.alphabet:
                    if letter not in guessed_letters:
                        letter_idx = self.letter_to_idx[letter]
                        letter_scores[letter] += hmm_scores[letter_idx] * hmm_weight
        
        # Strategy 3: Global frequency
        if global_weight > 0 and self.global_letter_freq:
            for letter in self.alphabet:
                if letter not in guessed_letters:
                    letter_scores[letter] += self.global_letter_freq[letter] * global_weight
        
        # Strategy 4: Context-aware (bigrams, first/last letter patterns)
        if context_weight > 0 and word_length in self.models_by_length:
            model = self.models_by_length[word_length]
            
            # Check for revealed letters and use bigram predictions
            for pos, char in enumerate(masked_word):
                if char != '_':
                    # Check next position
                    if pos + 1 < word_length and masked_word[pos + 1] == '_':
                        if char in model['letter_bigrams']:
                            for next_letter, prob in model['letter_bigrams'][char].items():
                                if next_letter not in guessed_letters:
                                    letter_scores[next_letter] += prob * context_weight * 0.5
                    
                    # Check previous position
                    if pos - 1 >= 0 and masked_word[pos - 1] == '_':
                        # Look for what commonly comes before this letter
                        for prev_letter in self.alphabet:
                            if prev_letter not in guessed_letters:
                                if prev_letter in model['letter_bigrams']:
                                    if char in model['letter_bigrams'][prev_letter]:
                                        letter_scores[prev_letter] += \
                                            model['letter_bigrams'][prev_letter][char] * context_weight * 0.5
            
            # First and last position boosts
            if masked_word[0] == '_' and self.common_first_letters:
                for letter, freq in self.common_first_letters.items():
                    if letter not in guessed_letters:
                        letter_scores[letter] += freq * context_weight * 0.3
            
            if masked_word[-1] == '_' and self.common_last_letters:
                for letter, freq in self.common_last_letters.items():
                    if letter not in guessed_letters:
                        letter_scores[letter] += freq * context_weight * 0.3
        
        # Vowel boost for early game
        if num_revealed <= 1:
            for vowel in self.vowels:
                if vowel not in guessed_letters:
                    letter_scores[vowel] *= 1.3
        
        # Convert to probabilities
        remaining_letters = {letter: score for letter, score in letter_scores.items() 
                           if letter not in guessed_letters and score > 0}
        
        if not remaining_letters:
            # Fallback to frequency-based guess
            remaining_letters = {letter: self.global_letter_freq.get(letter, 0.001) 
                               for letter in self.alphabet if letter not in guessed_letters}
        
        # Normalize
        total = sum(remaining_letters.values())
        if total > 0:
            remaining_letters = {k: v/total for k, v in remaining_letters.items()}
        
        return remaining_letters
    
    def get_best_guess(self, masked_word, guessed_letters):
        """Get single best letter guess"""
        probs = self.predict_letter_probabilities(masked_word, guessed_letters)
        if probs:
            return max(probs.items(), key=lambda x: x[1])[0]
        return None
    
    def get_top_letters(self, masked_word, guessed_letters, top_k=5):
        """Get top k most probable letters"""
        probs = self.predict_letter_probabilities(masked_word, guessed_letters)
        sorted_letters = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_letters[:top_k]
    
    def save_model(self, filename):
        """Save trained model"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'max_word_length': self.max_word_length,
                'models_by_length': self.models_by_length,
                'words_by_length': dict(self.words_by_length),
                'global_letter_freq': self.global_letter_freq,
                'common_first_letters': self.common_first_letters,
                'common_last_letters': self.common_last_letters
            }, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load trained model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.max_word_length = data['max_word_length']
            self.models_by_length = data['models_by_length']
            self.words_by_length = defaultdict(list, data['words_by_length'])
            self.global_letter_freq = data['global_letter_freq']
            self.common_first_letters = data['common_first_letters']
            self.common_last_letters = data['common_last_letters']
        print(f"Model loaded from {filename}")


def test_hmm_on_words(hmm, test_file, max_games=2000, verbose=True):
    if verbose:
        print("Testing HMM on test set")
        
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    test_words = [w for w in test_words if w.isalpha()][:max_games]
    
    total_games = len(test_words)
    wins = 0
    total_wrong = 0
    total_repeated = 0
    max_lives = 6
    
    game_history = []
    
    for i, target_word in enumerate(test_words):
        # Initialize game state
        masked_word = '_' * len(target_word)
        guessed_letters = set()
        wrong_guesses = 0
        repeated_guesses = 0
        lives_left = max_lives
        guess_sequence = []
        
        while lives_left > 0 and '_' in masked_word:
            # Get best guess
            best_letter = hmm.get_best_guess(masked_word, guessed_letters)
            
            if not best_letter:
                break
            
            # Check if repeated
            if best_letter in guessed_letters:
                repeated_guesses += 1
                continue
            
            guessed_letters.add(best_letter)
            guess_sequence.append(best_letter)
            
            # Check if letter is in word
            if best_letter in target_word:
                # Update masked word
                new_masked = list(masked_word)
                for idx, char in enumerate(target_word):
                    if char == best_letter:
                        new_masked[idx] = best_letter
                masked_word = ''.join(new_masked)
            else:
                wrong_guesses += 1
                lives_left -= 1
        
        # Check win condition
        won = '_' not in masked_word
        if won:
            wins += 1
        
        total_wrong += wrong_guesses
        total_repeated += repeated_guesses
        
        game_history.append({
            'word': target_word,
            'won': won,
            'wrong': wrong_guesses,
            'repeated': repeated_guesses,
            'guesses': guess_sequence
        })
        
        if verbose and (i + 1) % 100 == 0:
            current_rate = wins / (i + 1)
            print(f"Progress: {i+1}/{total_games} games | Win rate: {current_rate:.2%}")
    
    # Calculate metrics
    success_rate = wins / total_games
    avg_wrong = total_wrong / total_games
    avg_repeated = total_repeated / total_games
    final_score = (success_rate * total_games) - (total_wrong * 5) - (total_repeated * 2)
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Games: {total_games}")
        print(f"Wins: {wins}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Total Wrong Guesses: {total_wrong}")
        print(f"Avg Wrong Guesses per Game: {avg_wrong:.2f}")
        print(f"Total Repeated Guesses: {total_repeated}")
        print(f"Avg Repeated Guesses per Game: {avg_repeated:.2f}")
        print(f"\nFinal Score: {final_score:.2f}")
        print("="*60)
        
        # Show some successful games
        successful_games = [g for g in game_history if g['won']]
        failed_games = [g for g in game_history if not g['won']]
        
        if successful_games:
            print(f"\nSample of successful games (won with few mistakes):")
            sorted_wins = sorted(successful_games, key=lambda x: x['wrong'])[:5]
            for game in sorted_wins:
                print(f"  Word: {game['word']}, Wrong: {game['wrong']}, Guesses: {game['guesses'][:8]}")
        
        if failed_games:
            print(f"\nSample of failed games:")
            for game in failed_games[:5]:
                print(f"  Word: {game['word']}, Wrong: {game['wrong']}, Guesses: {game['guesses'][:10]}")
    
    return {
        'success_rate': success_rate,
        'total_wrong': total_wrong,
        'avg_wrong': avg_wrong,
        'total_repeated': total_repeated,
        'avg_repeated': avg_repeated,
        'final_score': final_score,
        'wins': wins,
        'total_games': total_games,
        'game_history': game_history
    }


# Main execution
if __name__ == "__main__":
    
    # Initialize and train HMM
    hmm = ImprovedHangmanHMM(max_word_length=30)
    
    # Train on corpus
    print("\nStep 1: Training HMM on corpus.txt")
    hmm.train('corpus.txt')
    
    # Save model
    hmm.save_model('improved_hmm_model.pkl')
    
    # Test on test set
    print("\nStep 2: Testing on test.txt")
    results = test_hmm_on_words(hmm, 'test.txt', max_games=2000)
    
    # Example usage
    print("EXAMPLE: Interactive Letter Prediction")
    example_word = "_pp_e"
    guessed = {'a', 'e', 'p'}
    top_letters = hmm.get_top_letters(example_word, guessed, top_k=5)
    print(f"Masked word: {example_word}")
    print(f"Already guessed: {guessed}")
    print(f"Top 5 letter predictions:")
    for i, (letter, prob) in enumerate(top_letters, 1):
        print(f"  {i}. {letter}: {prob:.4f}")