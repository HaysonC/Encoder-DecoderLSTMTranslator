from tensorflow.keras.models import Model, load_model
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model
model = load_model("seq2seq_model.h5")

from load_data import LoadData
# Load the tokenizers and other necessary data
with open('data.pkl', 'rb') as f:
    data: LoadData = pickle.load(f)

input_tokenizer = data.get('input_tokenizer')
output_tokenizer = data.get('output_tokenizer')
MAX_INPUT_LENGTH = data.get('MAX_INPUT_LENGTH')
MAX_OUTPUT_LENGTH = data.get('MAX_OUTPUT_LENGTH')

LATENT_DIM = 256  # Ensure this matches the model's latent dimension
NUM_LAYERS = 4  # Number of LSTM layers

def fetch_layer(layer_name: str) -> int:
    """
    Fetch a layer from the model by name

    Format: fetch_layer('lstm_1')
    :param layer_name:
    :return: the index of the layer
    """
    if layer_name.endswith("0"):
        layer_name = layer_name[:-2]
    for i, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return i
    raise ValueError(f"Layer {layer_name} not found in model")

print(model.layers)

# Extract Encoder Model
encoder_input = model.input[0]  # Encoder input
encoder_embedding = model.layers[fetch_layer('embedding')](encoder_input)
encoder_lstm = []
encoder_states = []
x = encoder_embedding
for i in range(NUM_LAYERS):
    return_state = (i == NUM_LAYERS - 1)  # Only return state for the last layer
    lstm_layer = model.layers[fetch_layer(f'lstm_{i}')]
    if return_state:
        x, state_h, state_c = lstm_layer(x)
        encoder_states = [state_h, state_c]
    else:
        x = lstm_layer(x)[0]
    encoder_lstm.append(lstm_layer)

# Define Decoder Model
decoder_input = model.input[1]  # Decoder input
decoder_embedding = model.layers[fetch_layer('embedding_1')](decoder_input)
decoder_lstm = []
x = decoder_embedding
for i in range(NUM_LAYERS):
    return_state = (i == NUM_LAYERS - 1)
    lstm_layer = model.layers[fetch_layer(f'lstm_{i + NUM_LAYERS}')]
    if return_state:
        x, _, _ = lstm_layer(x, initial_state=encoder_states)
    else:
        x = lstm_layer(x)[0]
    decoder_lstm.append(lstm_layer)
decoder_dense = model.layers[fetch_layer('dense')]
decoder_output = x

model = Model([encoder_input, decoder_input], decoder_output)
encoder_model = Model(encoder_input, encoder_states)
decoder_model = Model([decoder_input] + encoder_states, decoder_output)

print("Models loaded successfully")
print("Model Summary")
model.summary()


# Function to decode sequences
def decode_sequence(input_text):
    # Tokenize the input text
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=MAX_INPUT_LENGTH, padding='post')

    # Encode the input sequence to get the internal state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Populate the first character of target sequence with the start character
    target_seq[0, 0] = output_tokenizer.word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        decoder_res = decoder_model.predict([target_seq] + states_value)
        print(decoder_res)
        x, h, c = decoder_res
        output_tokens = decoder_dense(x)
        sampled_token_index = np.argmax(output_tokens)
        sampled_char = output_tokenizer.index_word[sampled_token_index]

        # Exit condition: either hit max length or find stop character
        if sampled_char == 'end' or len(decoded_sentence) > MAX_OUTPUT_LENGTH:
            stop_condition = True

        decoded_sentence += ' ' + sampled_char

        # Update the target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Test the Model
print(decode_sequence("Ich bin ein Student"))
print(decode_sequence("Hallo, wie geht es dir?"))
print(decode_sequence("Guten Morgen"))

# Interactive Chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = decode_sequence(user_input)
    print("Bot:", response)