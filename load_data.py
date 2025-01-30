import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import regex as re
from InquirerPy import inquirer

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
reverse_word_index_output: dict
input_tokenizer: Tokenizer
output_tokenizer: Tokenizer

# Load Europarl data

def load_europarl_data(file_path_en: str, file_path_de: str) -> tuple[list[str], list[str]]:
    """
    Loads Europarl data from two separate files: English and German.

    :param file_path_en: Path to the English file (.en)
    :param file_path_de: Path to the German file (.de)
    :return: A tuple containing lists of English and German sentences
    """
    with open(file_path_en, 'r', encoding='utf-8') as f_en, open(file_path_de, 'r', encoding='utf-8') as f_de:
        input_texts = [line.strip() for line in f_de.readlines()]
        target_texts = [line.strip() for line in f_en.readlines()]

    if len(input_texts) != len(target_texts):
        raise ValueError("The number of lines in the English and German files do not match.")

    print("Sample German Sentences:", input_texts[:5])
    print("Sample English Sentences:", target_texts[:5])

    return input_texts, target_texts

# Load TMX data

def load_tmx_data(file_path: str) -> tuple[list[str], list[str]]:
    input_texts = []
    target_texts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    german = False
    english = False
    print("Reading TMX data...")
    for line in tqdm(lines, desc="Processing lines"):
        line = line.strip()
        if line.startswith('<tu>'):
            german = True
        elif line.startswith('</tu>'):
            german = False
            english = False
        elif german:
            match = re.search(r'<seg>(.*)</seg>', line)
            if match:
                input_texts.append(match.group(1))
            german = False
            english = True

        elif english:
            match = re.search(r'<seg>(.*)</seg>', line)
            if match:
                target_texts.append(match.group(1))
            english = False

    print("TMX Data read successfully")
    print("Sample German Sentences:", input_texts[:5])
    print("Sample English Sentences:", target_texts[:5])
    return input_texts, target_texts

# Main function

def main():
    global INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH
    global encoder_input_train, decoder_input_train, decoder_target_train
    global encoder_input_val, decoder_input_val, decoder_target_val
    global reverse_word_index_output
    global input_tokenizer, output_tokenizer

    # Ask the user for the type of dataset using a dropdown
    dataset_type = inquirer.select(
        message="Select the dataset type:",
        choices=["tmx", "europarl"],
    ).execute()

    if dataset_type == 'tmx':
        while True:
            try:
                data_path = input("Enter the path to the TMX dataset: ").strip()
                input_texts, target_texts = load_tmx_data(data_path)
                break
            except FileNotFoundError:
                print("File not found. Please check the file path.")

    elif dataset_type == 'europarl':
        while True:
            try:
                file_path_de = input("Enter the path to the German file (.de): ").strip()
                file_path_en = input("Enter the path to the English file (.en): ").strip()
                input_texts, target_texts = load_europarl_data(file_path_en, file_path_de)
                break
            except FileNotFoundError:
                print("File not found. Please check the file path.")
    else:
        print("Invalid dataset type. Please enter 'tmx' or 'europarl'.")
        return

    # Add start and end tokens to target sentences for training
    target_texts = ['sos ' + text + ' eos' for text in target_texts]

    # Tokenize the input (German) and output (English) sentences with progress bar
    print("Tokenizing Input")
    input_tokenizer = Tokenizer()
    input_tokenizer.fit_on_texts(input_texts)
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)
    INPUT_VOCAB_SIZE = len(input_tokenizer.word_index) + 1

    print("Tokenizing Output")
    output_tokenizer = Tokenizer()
    output_tokenizer.fit_on_texts(target_texts)
    output_sequences = output_tokenizer.texts_to_sequences(target_texts)
    OUTPUT_VOCAB_SIZE = len(output_tokenizer.word_index) + 1

    # Get max lengths for padding
    MAX_INPUT_LENGTH = max(len(seq) for seq in input_sequences)
    MAX_OUTPUT_LENGTH = max(len(seq) for seq in output_sequences)

    # Pad the sequences
    encoder_input = pad_sequences(input_sequences, maxlen=MAX_INPUT_LENGTH, padding='post')
    decoder_input = pad_sequences(output_sequences, maxlen=MAX_OUTPUT_LENGTH, padding='post')

    # Create decoder target data (shifted by one timestep)
    decoder_target = np.zeros_like(decoder_input)
    decoder_target[:, :-1] = decoder_input[:, 1:]

    # Split data into training and validation sets
    encoder_input_train, encoder_input_val, decoder_input_train, decoder_input_val, decoder_target_train, decoder_target_val = train_test_split(
        encoder_input, decoder_input, decoder_target, test_size=0.2, random_state=42
    )

    reverse_word_index_output = dict([(value, key) for (key, value) in output_tokenizer.word_index.items()])
    print("Data processing complete.")

if __name__ == "__main__":
    main()
