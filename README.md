# HumanBiasInSemantics
Semantics derived automatically from language corpora contain human-like biases

1) Download the GloVe model to the code folder from: http://nlp.stanford.edu/data/glove.840B.300d.zip
The extracted model is a 5.65GB text file: "glove.840B.300d.txt".

2) Run "python3 convert_text_embeddings_to_binary.py glove.840B.300d.txt" to convert the word embeddings file (glove.840B.300d.txt) from .txt to binary for faster loading in the main program. This is a ONE time lengthy process, but will save you time later when running the tests. This step will result in creation of files "embedding.npy" and "embedding.vocab" which will be used for loading embeddings in WEAT and WEFAT tests.

3) Run "python3 test.py" for running different WEAT tests.

4) Run "python3 test1.py" for running WEFAT test.
