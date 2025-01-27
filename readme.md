# Sequence-to-Sequence Model with Two LSTMs

This repository contains a sequence-to-sequence model implemented using TensorFlow. The model is designed for tasks like machine translation and leverages two LSTM layers: one for encoding the input sequence and another for decoding the output sequence.

## Introduction to the Model

The model architecture includes:
1. **Encoder LSTM**: Processes the input sequence and encodes it into a fixed-dimensional vector.
2. **Decoder LSTM**: Decodes the vector to generate the target sequence.

The architecture supports variable-length input and output sequences, making it suitable for tasks such as translation or summarization.

## File Descriptions

1. `load_data.py`: Prepares the data for training and validation by tokenizing, padding, and splitting.
2. `sample.ipynb`: Contains a notebook demonstrating the training and evaluation of the model.
3. A standard tmx file is used for training and validation. You can download the file from OPUS: http://opus.nlpl.eu/
## Prerequisites

Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- numpy

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
Replace the ```load_data.main(str)``` call with the path to the tmx file you downloaded from OPUS. The file is used to train and validate the model.
Run the notebook `sample.ipynb` to train and evaluate the model.
