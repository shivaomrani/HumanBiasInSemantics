# code obtained from https://github.com/vered1986/PythonUtils/blob/master/word_embeddings/format_convertion/convert_text_embeddings_to_binary.py


from __future__ import print_function

import sys
import codecs
import numpy as np

from docopt import docopt

def main():

    embedding_file = sys.argv[1]
    print('Loading embeddings file from {}'.format(embedding_file))
    wv, words = load_embeddings(embedding_file)

    out_emb_file, out_vocab_file = embedding_file.replace('.txt', ''), embedding_file.replace('.txt', '.vocab')
    print('Saving binary file to {}'.format(out_emb_file))
    np.save(out_emb_file, wv)

    print('Saving vocabulary file to {}'.format(out_vocab_file))
    with codecs.open(out_vocab_file, 'w', 'utf-8') as f_out:
        for word in words:
            f_out.write(word + '\n')


def load_embeddings(file_name):
    """
    Load the pre-trained embeddings from a file
    :param file_name: the embeddings file
    :return: the vocabulary and the word vectors
    """
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        lines = [line.strip() for line in f_in]

    embedding_dim = len(lines[0].split()) - 1
    words, vectors = zip(*[line.strip().split(' ', 1) for line in lines if len(line.split()) == embedding_dim + 1])
    wv = np.loadtxt(vectors)

    return wv, words


main()
