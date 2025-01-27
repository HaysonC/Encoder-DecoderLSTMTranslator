import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

INPUT_VOCAB_SIZE: int
OUTPUT_VOCAB_SIZE: int
MAX_INPUT_LENGTH: int
MAX_OUTPUT_LENGTH: int
encoder_input_train: np.ndarray
decoder_input_train: np.ndarray
decoder_target_train: np.ndarray
encoder_input_val: np.ndarray
decoder_input_val: np.ndarray
decoder_target_val: np.ndarray

# Load the data (assuming it's in a text file with German and English sentences line by line)
def load_data(file_path):
    input_texts = []
    target_texts = []

    # Open the file and read line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # Loop through the lines, assuming German and English sentences are in pairs
        for i in range(0, len(lines) - 1, 2):  # Increment by 2 for each German-English pair
            german = lines[i].strip()  # Get the German sentence
            english = lines[i + 1].strip()  # Get the English sentence

            input_texts.append(german)
            target_texts.append(english)

    return input_texts, target_texts

def main(data_path: str) -> None:
    global INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH
    global encoder_input_train, decoder_input_train, decoder_target_train
    global encoder_input_val, decoder_input_val, decoder_target_val

    # Load and preprocess data
    input_texts, target_texts = load_data(data_path)

    # Add start and end tokens to target sentences for training (e.g., "<start>" and "<end>")
    target_texts = ['\t' + text + '\n' for text in target_texts]

    # Tokenize the input (German) and output (English) sentences
    input_tokenizer = Tokenizer()
    input_tokenizer.fit_on_texts(input_texts)
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)
    INPUT_VOCAB_SIZE = len(input_tokenizer.word_index) + 1  # Adding 1 for padding token

    target_tokenizer = Tokenizer()
    target_tokenizer.fit_on_texts(target_texts)
    target_sequences = target_tokenizer.texts_to_sequences(target_texts)
    OUTPUT_VOCAB_SIZE = len(target_tokenizer.word_index) + 1  # Adding 1 for padding token

    # Pad the sequences to the same length
    MAX_INPUT_LENGTH = max(len(seq) for seq in input_sequences)
    MAX_OUTPUT_LENGTH = max(len(seq) for seq in target_sequences)

    encoder_input_data = pad_sequences(input_sequences, maxlen=MAX_INPUT_LENGTH, padding='post')
    decoder_input_data = pad_sequences(target_sequences, maxlen=MAX_OUTPUT_LENGTH, padding='post')

    # Prepare the decoder target data (shifted by one timestep for teacher forcing)
    decoder_target_data = np.zeros_like(decoder_input_data)
    decoder_target_data[:, 0:-1] = decoder_input_data[:, 1:]

    # Split data into training and validation sets
    encoder_input_train, encoder_input_val, decoder_input_train, decoder_input_val, decoder_target_train, decoder_target_val = train_test_split(
        encoder_input_data, decoder_input_data, decoder_target_data, test_size=0.2)

    # You can now use encoder_input_train, decoder_input_train, and decoder_target_train for training
