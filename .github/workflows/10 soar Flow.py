#   intenseTensorFlow    #
#------------------------#
!pip install tensorflow
!pip install transformers
!pip install numpy
#------------------------#
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import random

def load_word_list():
    word_list = [
        'three', 'eons', 'long', 'ago', 'bye', 'barbeque', 'queen', 'quick',
        'quiet', 'quality', 'quintessential', 'question', 'queue', 'quantum',
        'frequent', 'antique', 'apple', 'banana', 'cat', 'dog', 'elephant',
        'zero', 'ocean', 'orbit', 'one', 'ice', 'island', 'two', 'echo', 'equal',
        'three', 'eons', 'earth', 'four', 'evil', 'edge', 'five', 'elf', 'energy',
        'six', 'exit', 'exact', 'seven', 'even', 'easy', 'eight', 'era', 'ever',
        'nine', 'end', 'entire', 'apple', 'ant', 'away', 'ago', 'ball', 'boy',
        'bye', 'blue', 'cat', 'car', 'can', 'cold', 'dog', 'day', 'did', 'dark',
        'egg', 'eye', 'eat', 'eons', 'fox', 'fan', 'for', 'fast', 'goat', 'game',
        'go', 'good', 'hat', 'house', 'has', 'hot', 'ice', 'ink', 'is', 'into',
        'juice', 'joy', 'just', 'jump', 'key', 'king', 'keep', 'kind', 'lion',
        'leg', 'long', 'like', 'man', 'map', 'may', 'more', 'nut', 'nose', 'now',
        'nice', 'owl', 'orange', 'on', 'old', 'pen', 'pig', 'put', 'pink', 'queen',
        'quick', 'quiet', 'quest', 'rat', 'run', 'red', 'round', 'sun', 'snake',
        'see', 'slow', 'tree', 'time', 'to', 'tall', 'umbrella', 'up', 'under',
        'use', 'van', 'voice', 'very', 'view', 'water', 'wind', 'was', 'with',
        'xylophone', 'xray', 'exit', 'extra', 'yellow', 'yes', 'you', 'young',
        'zebra', 'zoo', 'zero', 'zone'
    ]
    return word_list

def char_to_word(char, word_list):
    possible_words = [word for word in word_list if char in word]
    if not possible_words:
        return char
    return random.choice(possible_words)

def generate_sentence(model, char_to_idx, idx_to_char, essay, max_length=50):
    generated_text = essay[:]

    for _ in range(max_length):
        input_seq = [char_to_idx.get(c, 0) for c in generated_text]  # get the index for each char, 0 if not found
        input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_length)
        predicted_probs = model.predict(input_seq)[0]
        next_char_idx = np.argmax(predicted_probs)
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char
        if next_char == ' ':
            break

    return generated_text

def build_model(vocab_size, embedding_dim=256, lstm_units=512):
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        LSTM(lstm_units, return_sequences=True),
        LSTM(lstm_units),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def write_essay(input_string):
    word_list = load_word_list()
    words = [char_to_word(char, word_list) for char in input_string]
    essay = ' '.join(words)

    # Define characters and build the character index dictionaries
    # Include space in the characters
    characters = sorted(set(''.join(word_list) + ' '))  
    char_to_idx = {char: idx for idx, char in enumerate(characters)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Build and train the model (for simplicity, using a small set of epochs and synthetic data)
    vocab_size = len(characters)
    model = build_model(vocab_size)

    # Creating synthetic training data (for demonstration purposes)
    x_train = [[char_to_idx[char] for char in ''.join(word_list)]]
    y_train = [[char_to_idx[char] for char in ''.join(word_list)[1:] + ' ']]  # shifted by one position

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100, padding='post')
    y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train, maxlen=100, padding='post')

    # Train the model
    #model.fit(x_train, y_train, epochs=5)
    y_train = [[char_to_idx[char] for char in ''.join(word_list)[1:] + ' ']]  # shifted by one position

    # Generate a logical sentence
    logical_sentence = generate_sentence(model, char_to_idx, idx_to_char, essay)
    return logical_sentence

if __name__ == "__main__":
    input_string = input("\nAttach string here (up to 64 characters): ")
    input_string = input_string[:64]
    print(f"Input string: {input_string}")

    essay = write_essay(input_string)
    print(f"\n{essay}\n")
