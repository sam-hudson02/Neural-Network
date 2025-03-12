import json
import nltk
from collections import defaultdict
# from utils.activation import ReLU, Sigmoid, Tanh
# from layers.activation import Activation as ActivationLayer
# from layers.convolution import Convolution
# from layers.conv1d import Conv1d
# from layers.maxpool import MaxPool
# from layers.layer import Layer
# from layers.reshape import Reshape
# from layers.flatten import Flatten
# from plots.plot_net import plot_net
# from utils.optimizer import Optimizers
# from utils.utils import cce, cce_softmax_prime
# from layers.dense import Dense
# from models.nn import Network
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from alive_progress import alive_bar
import numpy as np
import random
nltk.download('punkt_tab')

classes = [
    "History", "Fantasy", "Drama", "Mystery", "Science fiction", "Romance",
    "Fiction, coming of age", "Fiction, horror", "Fiction, action & adventure",
]


def open_book_data(path: str):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def create_classes():
    max = 0
    i_max = 0
    path = './data/books/filtered_books.json'
    book_json = open_book_data(path)
    new_book_json = []
    str_desc = []
    for i, book in enumerate(book_json):
        subjects = book['subjects']
        mulit_label = []
        for c in classes:
            if c in subjects:
                book['class'] = classes.index(c)
                mulit_label.append(1)
            else:
                mulit_label.append(0)
        book['multi'] = mulit_label
        desc = book['description']
        if type(desc) is str:
            str_desc.append(i)
            desc = {'type': '/type/text', 'value': desc}
        book['description'] = desc
        text = desc['value']
        words = text.split()
        if len(words) <= 500:
            new_book_json.append(book)
            if len(words) > max:
                max = len(words)
                i_max = i

    print(max)
    print(i_max)
    max_desc = book_json[i_max]['description']['type']
    print(max_desc)
    meta = {}
    print(i_max in str_desc)
    title = book_json[i_max]['title']
    print(title)
    print(len(book_json))
    print(len(new_book_json))

    for i, c in enumerate(classes):
        meta[i] = c

    save_json(new_book_json, './data/books/books.json')
    save_json(meta, './data/books/meta.json')


def tokenize():
    path = './data/books/books.json'
    book_json = open_book_data(path)
    word_count = defaultdict(int)
    word_in_doc = defaultdict(set)
    new_book_json = []
    total = 0
    for book in book_json:
        idf_count = defaultdict(int)
        text = book['description']['value'].lower()
        sentences = nltk.sent_tokenize(text)
        toks = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            work_id = book['work_id']
            for word in words:
                word_count[word] += 1
                idf_count[word] += 1
                total += 1
                word_in_doc[word].add(work_id)
            toks.append(words)
        book['count'] = idf_count
        book['tokens'] = toks
        new_book_json.append(book)
    new_book_json = tf_idf(new_book_json, word_in_doc)
    save_json(new_book_json, './data/books/books.json')
    vocab = list(word_count.keys())
    save_json(vocab, './data/books/vocab.json')


def tf_idf(book_json: list, word_in_doc: dict):
    total_docs = len(book_json)
    new_book_json = []
    for book in book_json:
        tfidf = {}
        unique_tokens = list(book['count'].keys())
        len_tokens = 0
        for sent in book['tokens']:
            len_tokens += len(sent)
        for token in unique_tokens:
            t_in_doc = book['count'][token]
            tf = t_in_doc / len_tokens
            doc_containing_d = len(word_in_doc[token])
            idf = np.log(total_docs / doc_containing_d)
            tfidf[token] = tf * idf
        book['tf-idf'] = tfidf
        new_book_json.append(book)
    return new_book_json


def make_skipgrams():
    path = './data/books/books.json'
    book_json = open_book_data(path)
    total_bigrams = []
    for book in book_json:
        toks = book['tokens']
        for sent in toks:
            bigrams = list(nltk.bigrams(sent))
            total_bigrams += bigrams
    return total_bigrams


def check():
    book_json = open_book_data('./data/books/books.json')
    print(book_json[0])
    data = open_book_data('./data/books/data.json')
    print(data['x'][:10])
    print(data['y'][:10])
    print(len(data['x']))
    print(len(data['y']))


def make_x_y():
    print('creating skipgrams')
    skipgrams = list(set(make_skipgrams()))
    print(skipgrams[:1000])
    print(f'made {len(skipgrams)} skipgrams')
    print('loading vocab')
    vocab = open_book_data('./data/books/vocab.json')
    word_ind = {word: i for i, word in enumerate(vocab)}
    print('creating x list')
    x_list = [word_ind[skipgram[0]] for skipgram in skipgrams]
    print('creating y list')
    y_list = [word_ind[skipgram[1]] for skipgram in skipgrams]
    data = {'x': x_list, 'y': y_list}
    print('saving data')
    save_json(data, './data/books/data.json')
    return x_list, y_list


def word2vec():
    data = open_book_data('./data/books/data.json')
    vocab = open_book_data('./data/books/vocab.json')
    x_list = data['x'][0:32000]
    y_list = data['y'][0:32000]
    x = one_hot(x_list, vocab)
    y = one_hot(y_list, vocab)

    x_train = x[:-2000]
    y_train = y[:-2000]
    x_test = x[-2000:]
    y_test = y[-2000:]
    embed_size = 128
    vec_size = x.shape[1]
    opt = Optimizers.ADAM
    layers: list[Layer] = [
        Dense(vec_size, embed_size, opt),
        Dense(embed_size, vec_size, opt),
    ]

    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    val = (x_test, y_test)
    network.train(x_train, y_train, epochs=10, validation=val, batch_size=128)
    network.save('./models/word2vec')
    loss = network.loss_history
    print(loss[0][:20])
    loss = network.average_loss()
    acc = network.average_accuracy()
    val_loss = network.validation_loss_history
    val_acc = network.validation_accuracy_history
    plot_net(loss, acc, val_loss, val_acc)


def open_word2vec():
    embed_size = 128
    vec_size = 1
    opt = Optimizers.ADAM
    layers: list[Layer] = [
        Dense(1, embed_size, opt),
        Dense(embed_size, vec_size, opt),
    ]

    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)
    network.open('./models/word2vec')
    loss = network.average_loss()
    acc = network.average_accuracy()
    val_loss = network.validation_loss_history
    val_acc = network.validation_accuracy_history
    plot_net(loss, acc, val_loss, val_acc)


def doc2vec():
    books = open_book_data('./data/books/books.json')
    data = []
    for book in books:
        total = []
        for sent in book['tokens']:
            total += sent
        data.append(total)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
    model = Doc2Vec(documents, vector_size=256,
                    window=4, min_count=1, workers=6, epochs=30)
    model.save('./models/doc2vec')
    with alive_bar(len(data)) as bar:
        for i, doc in enumerate(data):
            vector = model.infer_vector(doc)
            books[i]['vector'] = list(vector.astype(float))
            bar()
    save_json(books, './data/books/books.json')


def sbert():
    books = open_book_data('./data/books/books.json')
    data = []
    for book in books:
        data.append(book['description']['value'])
    model = SentenceTransformer("all-distilroberta-v1")
    embeddings = model.encode(data)
    for i, book in enumerate(books):
        books[i]['sbert'] = list(embeddings[i].astype(float))
    save_json(books, './data/books/books.json')


def one_hot(data: list, classes: int):
    x = np.zeros((len(data), classes))
    for i, word in enumerate(data):
        x[i, word] = 1
    return x


def create_train_test(split: float = 0.8):
    # open book data
    books: list = open_book_data('./data/books/books.json')
    count = defaultdict(int)
    for book in books[:100]:
        multi = book['multi']
        for i, c in enumerate(multi):
            print(multi)
            if c == 1:
                class_name = classes[i]
                count[class_name] += 1
    # randomize
    random.shuffle(books)
    # split
    split = int(len(books) * split)
    train = books[:split]
    test = books[split:]
    # save
    save_json(count, './data/books/count.json')
    save_json(train, './data/books/train.json')
    save_json(test, './data/books/test.json')


def load_data_old(sbert=False, normalize=False):
    books = open_book_data('./data/books/books.json')
    x = []
    y = []
    y_multi = []
    count = defaultdict(int)
    d_key = 'sbert' if sbert else 'vector'
    for book in books:
        if normalize:
            if count[book['class']] > 2000:
                continue
        x.append(np.array(book[d_key]))
        y.append(book['class'])
        y_multi.append(np.array(book['multi']))
        count[book['class']] += 1
    print(count)
    y = one_hot(y, len(classes))
    x = np.array(x)
    # shuffle
    perm = np.random.permutation(len(x))
    x = x[perm]
    y = y[perm]
    y_multi = np.array(y_multi)
    y_multi = y_multi[perm]
    x_train = x[:-1000]
    y_train = y[:-1000]
    y_multi_train = y_multi[:-1000]
    x_test = x[-1000:]
    y_test = y[-1000:]
    y_multi_test = y_multi[-1000:]

    return x_train, y_train, y_multi_train, x_test, y_test, y_multi_test


def load_data(sbert=False, normalize=False):
    x_train, y_train, y_multi_train = process_file(
        './data/books/train.json', normalize, sbert)
    x_test, y_test, y_multi_test = process_file(
        './data/books/test.json', normalize, sbert)

    return x_train, y_train, y_multi_train, x_test, y_test, y_multi_test


def process_file(path: str, normalize: bool = False, sbert: bool = False):
    data = open_book_data(path)
    x = []
    y = []
    y_multi = []
    count = defaultdict(int)
    d_key = 'sbert' if sbert else 'vector'
    for book in data:
        if normalize:
            if count[book['class']] > 2000:
                continue
        x.append(np.array(book[d_key]))
        y.append(book['class'])
        y_multi.append(np.array(book['multi']))
        for i, c in enumerate(book['multi']):
            if c == 1:
                count[i] += 1
    print(count)
    x = np.array(x)
    y = np.array(y)
    y_multi = np.array(y_multi)
    return x, y, y_multi


def books_model():
    x_train, y_train, x_test, y_test = load_data(sbert=True)
    h = 32
    w = 24
    x_train = x_train.reshape(-1, 1, h, w)
    x_test = x_test.reshape(-1, 1, h, w)
    vec_size = x_train.shape[1]
    # x_train = x_train.reshape(-1, 1, vec_size)
    # x_test = x_test.reshape(-1, 1, vec_size)
    opt = Optimizers.ADAM
    classes = y_train.shape[1]
    opt = Optimizers.ADAM
    layers: list[Layer] = [
        Dense(vec_size, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, classes, opt),
    ]
    filters_1 = 6
    filters_2 = 12
    l_2_h = (h // 2) - 1
    l_2_w = (w // 2) - 1
    l_3_h = l_2_h - 2
    l_3_w = l_2_w - 2
    layers = [
        Convolution((h, w, 1), filters_1, (3, 3), opt),
        ActivationLayer(ReLU()),
        MaxPool(2, 2),
        Convolution((l_2_h, l_2_w, filters_1), filters_2, (3, 3), opt),
        ActivationLayer(ReLU()),
        Reshape((l_3_w, l_3_h, filters_2)),
        Dense(l_3_h * l_3_w * filters_2, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, classes, opt),
    ]

    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)
    val = (x_test, y_test)
    print(x_train.shape)
    print(y_train.shape)
    network.train(x_train, y_train, epochs=100, validation=val, batch_size=128)
    network.save('./models/books_3')
    loss = network.average_loss()
    acc = network.average_accuracy()
    val_loss = network.validation_loss_history
    val_acc = network.validation_accuracy_history
    plot_net(loss, acc, val_loss, val_acc, 'books2')


if __name__ == '__main__':
    create_train_test()
