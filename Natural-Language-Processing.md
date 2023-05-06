# Natural Language Processing

## Contents

1. Tokenizer
2. Text to Sequence 
3. Padding
4. Out-of-Vocabulary Tokens 
5. Word Embedding
6. Long Short-Term Memory
7. CNN For NLP 
8. Layer Comparison
9. Generating Text

## Tokenizer

Extract vocabulary of words from corpus and represent the texts into numerical representations which can be used to train neural network.

`from tensorflow.keras.preprocessing.text import Tokenizer`

- Methods
1. `fit_on_texts()`: Generate indices for each word in the corpus 
2. `tokenizer.word_index` : Get the indices

By default all punctuations are ignored and words are converted to lower case. 

## Text to Sequences

Convert each of the input sentences into a sequence of tokens.

`sequences = tokenizer.text_to_sequences()`

## Padding

`from tensorflow.keras.preprocessing.sequence import pad_sequences`

We need to pad the sequences into a uniform length as the model expects the padded sequences. By default, it will pad according to the length of the longest sequence. We can override this with the `maxlen` argument to define a specific length.

`padded = pad_sequences(sentences, maxlen=_)`

## Out-of-Vocabulary Tokens 

This is used when we have words that are not found in the `word_index` dictionary. Token 1 is inserted for words that are not found in the dictionary.

## Word Embedding

The Embedding layer represents each word in the vocabulary with vectors. These vectors have trainable weights so as our neural network learns, words that are most likely to appear together will converge towards similar weights.

`tensorflow.keras.layers.Embedding(vocab_size, embedding_dim, input_length)`

Word embeddings can be visualized and can be plotted using the Tensorflow Embedding Projector.

It is useful to have `reverse_word_index` dictionary so we can quickly lookup a word based on a given index. 

`reverse_word_index = tokenizer.word_index`

- One important step in preprocessing the text data is to convert the labels inot numpy array.

We can use `GlobalAveragePooling1D` layer intead of `Flatten` after the `Embedding` layer. This adds the task of averaging over the sequence dimension before connecting to the dense layer.

> We can set the `verbose` parameter of `model.fit()` to 2 to indicate that we want to print just the results per epoch. Setting it to 1(default) displays a progress bar per epoch, while 0 silences all displays. In production `verbose` 2 is recommended.

We can use pre-tokenized dataset that is using subword text encoding, thsi is an alternative to word-based tokenization. Subwords tokenization eliminated the probles caused by the `<OOV>`, which increases the `vocab_size` ,slows down the training and bloats the model. Subword text encoding gets around this problem by using parts of the word to compose whole words. This makes it more flexible when it encounters uncommon words.

## Long Short-Term Memory

LSTM computes the state of a current timestep and passes it on to the next timestep where this state is also updated. The process repeats until the final timestep where the output computation is affected by all previous states. It can also be configured to be Bidirectional so we can get the relationship of later words to earlier ones. 

`tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(lstm_dim))`

We can build multiple layer of LSTM models by simply appending another LSTM layer in the model and enabling the `return_sequences` flag to `True`.

## CNN For NLP 

Convolutional neural networks can be used to solve NLP problems. For temporal data such as sequences, we can use the `Conv1D` model so the convolution will happen over a single dimension. We can also append a pooling layer to reduce the output of the convolution layer. 

## Layer Comparison

1. `Flatten`: Its main advantage is that it is very fast to train.
2. `LSTM`: This is slower to train but useful in applications where the order of the tokens is important.
3. `GRU`: A simpler version of LSTM. It can be used in applications where the sequence is important but we want faster results and can sacrifice some accuracy.
4. `Convolution`: Trains faster tha `LSTM` and `GRU`

## Generating Text 

Here the training text data acts both as input and label, the current word is treated as input and the next one as label, so on and so on.
The labesl are one-hot encoded arrays. The last output layer in the model is a `Dense` layer which uses the `softmax` activation functions and takes `total_words` as the input. The output layer will have one neuron for each word in the vocabulary. So given an input token list, the output array of the final layer will have the probabilities for each word.

- Process of generating text

1. Feed a seed text to initiate the process.
2. Model predicts the index of the most probable next word.
3. Look up the index in the reverse word index dictionary.
4. Append the next word to the seed text.
5. Feed the result to the model again.

Steps 2 to 5 will repeat until the desired length of the output is reached.

If there are a lot of reapeating of words in the output we can get the top three indices and choose one at random instead of getting the index with max probabilities.


