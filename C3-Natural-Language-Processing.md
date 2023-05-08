# Natural Language Processing

## Contents

1. Tokenizer
2. Padding
3. Word Embedding
4. Long Short-Term Memory
5. CNN For NLP 
6. Layer Comparison
7. Text Generation

### Tokenizer

Tokenizer is used to extract vocabulary of words from corpus and represent the texts into numerical representations which can be used to train neural network.

First of all we instantiate a Tokenizer by specifying the maximum number of words to index and a out of vocabulary token. The OOV token should be such that it does not appear by itself in the input text.

Then we extract and index the words from training sentences by calling `fit_on_texts()`. The resulting word index can be accessed using `word_index` property of tokenizer. Next we generate sequences (numeric array) from the sentences by calling `text_to_sequences()`. By default all punctuations are ignored and words are converted to lower case. 

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words = 10000, oov_token='<OOV>')

# Generate the word index dictionary for the training sentences
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Generate and pad the training sequences
sequences = tokenizer.texts_to_sequences(training_sentences)
```

### Padding

Neural networks generally need all input data to be in uniform dimensions. Our sentences and sequences are of different lengths, so we need to make them uniform. For this, we pad the sequences into a uniform length using `pad_sequences` method. By default, it will pad according to the length of the longest sequence. We can override this with the `maxlen` argument to define a specific length.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences,
                       padding='pre'         # whether to pad at beginning (pre) or end(post), default pre
                       value=0.0             # value to use for padding, defaults to zero
                       maxlen=max_length,    # If value is provided longer sequences are truncated
                       truncating='pre'      # whether to truncate from begining or end
                      )
```

The sequences can be reversed using `sequences_to_text()` for inspection.

### Word Embedding

The Embedding layer represents each word in the vocabulary with vectors. These vectors have trainable weights so as our neural network learns, the vectors for the words that are most likely to appear together will orient in the same direction.

We can use `GlobalAveragePooling1D` layer intead of `Flatten` after the `Embedding` layer. This adds the task of averaging over the sequence dimension before connecting to the dense layer.

```python
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length)
    tf.keras.layers.GlobalAveragePooling1D(),
```

Word embeddings can be visualized and can be plotted using the [Tensorflow Embedding Projector](https://projector.tensorflow.org/).

### Subword Tokenization

In word tokenization, out of vocab words are a big issue as we need a huge index to cover all words in training data. This increases the `vocab_size`, slows down the training and bloats the model. Even then there may be words in validation or test set that were not present in training data and result in out of vocab index.

As an alternative, can use a pre-tokenized dataset that uses subword text encoding. Subword text encoding gets around the out of vocab problem by using parts of the word to compose whole words. This makes it more flexible when it encounters uncommon words.

### Long Short-Term Memory

LSTM is useful where the order of tokens is important. LSTM computes the state of a current timestep and passes it on to the next timestep where this state is also updated. The process repeats until the final timestep where the output computation is affected by all previous states. 

It can also be configured to be Bidirectional so we can get the relationship of later words to earlier ones. 

```python
tensorflow.keras.layers.LSTM(lstm_dim, return_sequences=True)
tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(lstm_dim))
```

We can chain multiple layer of LSTM models by simply appending another LSTM layer in the model and enabling the `return_sequences` flag to `True` in initial layers. 

### CNN For NLP 

Convolutional neural networks can be used to solve NLP problems. For temporal data such as text sequences, we can use the `Conv1D` model so the convolution will happen over a single dimension. We can also append a pooling layer to reduce the output of the convolution layer. 

```python
    tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
])
```

### Layer Comparison

1. `Flatten`: Flattens the input into single dimension output. Its main advantage is that it is very fast to train.
2. `LSTM`: This is slower to train but useful in applications where the order of the tokens is important.
3. `GRU`: A simpler version of LSTM. It can be used in applications where the sequence is important but we want faster results and can sacrifice some accuracy.
4. `Convolution`: Trains faster tha `LSTM` and `GRU`

### Generating Text 

We can use neural networks to generate text by training them to predict the next word based on previous words. Here the training text data acts both as input and label, the current word is treated as input and the next one as label, so on and so on. This can be understood using the following example:

The sentence `I am learning neural networks.` can be split into following sequnces where the last sequence is the label.

I | am
I | am | learning
I | am | learning | neural
I | am | learning | neural | networks

For doing the same programatically, first we take the input text and genrate sequences from it. Then for each sequence, we generate N-gram sequences which are left-padded. Then we split the sequences into input and label by using the last sequence as label.


```phython
# Initialize the sequences list
input_sequences = []

# Loop over every line
for line in corpus:

	# Tokenize the current line
	token_list = tokenizer.texts_to_sequences([line])[0]

	# Loop over the line several times to generate the subphrases
	for i in range(1, len(token_list)):
		
		# Generate the subphrase
		n_gram_sequence = token_list[:i+1]

		# Append the subphrase to the sequences list
		input_sequences.append(n_gram_sequence)

# Get the length of the longest line
max_sequence_len = max([len(x) for x in input_sequences])

# Pad all sequences
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create inputs and label by splitting the last token in the subphrases
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

# Convert the label into one-hot arrays
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

```

The output labels are one-hot encoded. The last output layer in the model is a `Dense` layer which uses the `softmax` activation functions and takes `total_words` as the input. The output layer will have one neuron for each word in the vocabulary. So given an input token list, the output array of the final layer will have the probabilities for each word.


