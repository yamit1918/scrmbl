#  touchy.Torchy #
# -------------- #
import random
from transformers import pipeline
import torch
from google.colab import userdata
userdata.get('UR-KEY')
def load_word_list():
    # Expanded list of words to choose from
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
    # Filter words containing the character
    possible_words = [word for word in word_list if char in word]
    if not possible_words:
        return char  # If no words contain the character, return the character itself
    return random.choice(possible_words)

def generate_sentence(essay):
   try:
        generator = pipeline('text-generation', model='gpt2')
        result = generator(essay, max_length=50, num_return_sequences=1)
        return result[0]['generated_text']
   except Exception as e:
      return f"Error generating sentence: {str(e)}"

def write_essay(input_string):
    word_list = load_word_list()
    words = [char_to_word(char, word_list) for char in input_string]

    # Add conjunctions and other words to make the essay more natural
    conjunctions = ['and', 'or', 'but', 'so', 'because', 'yet', 'for']
    essay = []

    for i, word in enumerate(words):
        essay.append(word)
        if i < len(words) - 1:
            essay.append(random.choice(conjunctions))

    # Generate a logical sentence using the language model
    logical_sentence = generate_sentence(' '.join(essay))
    return logical_sentence

# Main function to run the program
if __name__ == "__main__":
    input_string = input("\nAttach string here (up to 64 characters): ")
    input_string = input_string[:64]  # Ensure the string is up to 64 characters
    print(f"Input string: {input_string}")

    essay = write_essay(input_string)
    print(f"\n", essay, "\n")
