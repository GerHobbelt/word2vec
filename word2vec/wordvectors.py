from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import struct
try:
    from sklearn.externals import joblib
except:
    joblib = None

from word2vec.utils import unitvec


class WordVectors(object):

    def __init__(self, vocab, vectors, clusters=None, train=None):
        """
        Initialize a WordVectors class based on vocabulary and vectors

        This initializer precomputes the vectors of the vectors

        Parameters
        ----------
        vocab : np.array
            1d array with the vocabulary
        vectors : np.array
            2d array with the vectors calculated by word2vec
        clusters : word2vec.WordClusters (optional)
            1d array with the clusters calculated by word2vec
        """
        self.vocab = vocab
        self.vectors = vectors
        self.clusters = clusters
        self._buildIndexMap(vocab)
        self.train = train
        self.hidden_words = None
        self.hword_len = 0
        self.set_hidden_words()

    def ix(self, word):
        """
        Returns the index on self.vocab and `self.vectors` for `word`
        """
        if word not in self.index_map:
            raise KeyError('Word not in vocabulary')
        else:
            return self.index_map[word]

    def __getitem__(self, word):
        return self.get_vector(word)

    def __contains__(self, word):
        return word in self.index_map

    def _buildIndexMap(self, vocab):
        self.index_map = {}
        for idx, word in enumerate(vocab):
            self.index_map[word] = idx

    def evaluateSentence(self, sentence):
        if not self.train: return none
        
        word_list = sentence.split() #prune for valid words
        layer1_size = self.train['layer1_size']
        window = self.train['window']
        sentence_position = 0
        sentence_len = len(word_list)
        word = word_list[sentence_position]
        sentence_position = 0
        sum_error = 0
        for sentence_position in xrange(sentence_len):
            word = word_list[sentence_position]
            if word not in self:
                continue
            
            neu1 = np.zeros(layer1_size)
            neu1e = np.zeros(layer1_size)
            b = window
            curr_sum = 0
            curr_c = 0
            if self.train['cbow']:
                cw = 0
                for a in xrange(0, window * 1 + 1):
                    c = sentence_position - window + a
                    if c < 0: continue
                    if c >= sentence_len: continue
                    curr_word = word_list[c]
                    if curr_word not in self: continue
                    neu1 += self.get_vector(curr_word)
                    cw += 1

                if cw:
                    neu1 /= cw
                    grad_sum = 0
                    gc = 0
                    if self.train['hs']:
                        cw = 0
                        for d in xrange(self.train["vocab"][word]["codelen"]):
                            f = 0
                            l2 = self.train["vocab"][word]["point"][d]
                            f = 1. (1 + exp(np.dot(neu1, self.train['syn1'][l2])))
                            g = (1 - self.train['vocab'][word]['code'][d] - f)
                            grad_sum += abs(g)
                            gc += 1

                        curr_sum += grad_sum / gc
                        curr_c += 1
                    else:
                        for d in xrange(max([3, self.train['neg']]) + 1):
                            if d == 0:
                                target = word
                                label = 1
                            else:
                                target = self.vocab[np.random.randint(0, self.train['syn1_size'])]
                                if target ==  word: continue
                                label = 0
                            
                            l2 = target
                            f = 0
                            f = np.dot(neu1, self.train["syn1"][l2])
                            g = (label - 1. / (1 + np.exp(f)))
                            grad_sum += abs(g)
                            gc += 1

                        curr_sum += grad_sum / gc
                        curr_c += 1
            else:
                for a in xrange(0, window * 2 + 1):
                    c = sentence_position - window + a
                    if (a >= window * 2 + 1 - b): c =0
                    if c < 0: continue
                    if c >= sentence_len: continue
                    last_word =  word_list[c]
                    l1 = last_word 
                    neu1e = np.zeros(layer1_size)
                    grad_sum = 0
                    gc = 0
                    if self.train['hs']:
                        for d in xrange(self.train['vocab'][word]['codelen']):
                            f = 0
                            l2 = self.train['vocab'][word]['point'][d]
                            f += 1. / (1 - np.exp( np.dot(self.get_vector(last_word), self.train['syn1'][l2])))
                            g = (1 - self.train['vocab'][word]['code'][d] - f)
                            grad_sum += abs(g)
                            gc += 1

                        curr_sum += grad_sum / gc
                        curr_c += 1
                    else:
                        for d in xrange(max([3, self.train['neg']]) + 1):
                            if d == 0:
                                target = word
                                label = 1
                            else:
                                #target = self.vocab[np.random.randint(0, self.train['syn1_size'])]
                                target = self.hidden_words[np.random.randint(0, self.hword_len)]
                                if target ==  word: continue
                                label = 0
                            
                            l2 = target
                            f = 0
                            f = 1. /(1. + np.exp(np.dot(self.get_vector(last_word), self.train['syn1'][l2])))
                            g = (label - f)
                            grad_sum += abs(g)
                            gc += 1
                        
                        curr_sum += grad_sum / gc
                        curr_c += 1

            if curr_c > 0: sum_error += curr_sum / curr_c
        
        return sum_error / sentence_len
    
    def trainSentence(self, sentence, epochs=50, alpha=0.05):
        if not self.train: return None
        
        np.random.seed(42)
        word_list = sentence.split() #prune for valid words
        layer1_size = self.train['layer1_size']
        window = self.train['window']
        newvec =  np.random.random(layer1_size)
        sentence_position = 0
        sentence_len = len(word_list)
        word = word_list[sentence_position]
        step_size = (alpha - min(0.0001, alpha)) / (sentence_len * epochs)
        for epoch in xrange(epochs):
            sentence_position = 0
            for sentence_position in xrange(sentence_len):
                word = word_list[sentence_position]
                if word not in self: 
                    sentence_position += 1
                    alpha -= step_size
                    if sentence_position >= sentence_len:
                        break
                    continue
                neu1 = np.zeros(layer1_size)
                neu1e = np.zeros(layer1_size)
                b = np.random.randint(0, window)
                if self.train['cbow']:
                    cw = 0
                    for a in xrange(b, window * 1 + 1 - b):
                        c = sentence_position - window + a
                        if c < 0: continue
                        if c >= sentence_len: continue
                        curr_word = word_list[c]
                        if curr_word not in self: continue
                        neu1 += self.get_vector(curr_word)
                        cw += 1
                    
                    neu1 += newvec
                    cw += 1
                    if cw:
                        neu1 /= cw
                        if self.train['hs']:
                            for d in xrange(self.train["vocab"][word]["codelen"]):
                                f = 0
                                l2 = self.train["vocab"][word]["point"][d]
                                f = 1. / (1 + exp(np.clip(np.dot(neu1, self.train['syn1'][l2]),-3., 3.)))
                                g = (1 - self.train['vocab'][word]['code'][d] - f) * alpha
                                neu1e += g * self.train['syn1'][l2]
                        else:
                            for d in xrange(max([3, self.train['neg']]) + 1):
                                if d == 0:
                                    target = word
                                    label = 1
                                else:
                                    target = self.vocab[np.random.randint(0, self.train['syn1_size'])]
                                    #target = self.hidden_words[np.random.randint(0, self.hword_len)]
                                    if target ==  word: continue
                                    label = 0
                                
                                l2 = target
                                f = 0
                                f = np.clip(np.dot(neu1, self.train["syn1"][l2]),-3., 3.)
                                g = (label - 1. / (1 + np.exp(f))) * alpha
                                neu1e += g * self.train["syn1"][l2]
                        
                        newvec += neu1e
                else: #adjusting the tast word logic of this sequence to only learn the sentence embedding
                    if True:
                    #for a in xrange(b, window * 2 + 2 - b):
                        """c = sentence_position - window + a
                        if (a >= window * 2 + 1 - b): c =0
                        if c < 0: continue
                        if c >= sentence_len: continue
                        last_word =  word_list[c]
                        l1 = last_word 
                        """
                        neu1e = np.zeros(layer1_size)
                        if self.train['hs']:
                            for d in xrange(self.train['vocab'][word]['codelen']):
                                f = 0
                                l2 = self.train['vocab'][word]['point'][d]
                                #f += 1. / (1 - np.exp( np.dot(self.get_vector(last_word), self.train['syn1'][l2])))
                                f += 1. / (1 - np.exp( np.clip(np.dot(newvec, self.train['syn1'][l2]), -3., 3.)))
                                g = (1 - self.train['vocab'][word]['code'][d] - f) * alpha
                                neu1e += g * self.train['syn1'][l2]
                        else:
                            for d in xrange(max([3, self.train['neg']]) + 1):
                                if d == 0:
                                    target = word
                                    label = 1
                                else:
                                    target = self.vocab[np.random.randint(0, self.train['syn1_size'])]
                                    #target = self.hidden_words[np.random.randint(0, self.hword_len)]
                                    if target ==  word: continue
                                    label = 0
                                
                                l2 = target
                                f = 0
                                #f = 1. /(1. + np.exp(np.dot(self.get_vocab(last_word), self.train['syn1'][l2])))
                                f = 1. /(1. + np.exp(np.clip(np.dot(newvec, self.train['syn1'][l2]), -3., 3.)))
                                g = (label - f) * alpha
                                neu1e += g * self.train['syn1'][l2]
                            
                            newvec += neu1e
                            #comment out original logic
                            #model.get_vector(last_word) += neu1e
                
                sentence_position += 1
                alpha -= step_size
                if sentence_position >= sentence_len:
                    break
        
        return newvec
        
    def get_vector(self, word):
        """
        Returns the (vectors) vector for `word` in the vocabulary
        """
        idx = self.ix(word)
        return self.vectors[idx]

    def cosine(self, word, n=10):
        """
        Cosine similarity.

        metric = dot(vectors_of_vectors, vectors_of_target_vector)
        Uses a precomputed vectors of the vectors

        Parameters
        ----------
        word : string
        n : int, optional (default 10)
            number of neighbors to return

        Returns
        -------
        2 numpy.array:
            1. position in self.vocab
            2. cosine similarity
        """
        metrics = np.dot(self.vectors, self[word].T)
        best = np.argsort(metrics)[::-1][1:n+1]
        best_metrics = metrics[best]
        return best, best_metrics

    def cosine_vec(self, vec, n=10):
        """
        Cosine similarity.

        metric = dot(vectors_of_vectors, vectors_of_target_vector)
        Uses a precomputed vectors of the vectors

        Parameters
        ----------
        word : string
        n : int, optional (default 10)
            number of neighbors to return

        Returns
        -------
        2 numpy.array:
            1. position in self.vocab
            2. cosine similarity
        """
        metrics = np.dot(self.vectors, vec.T)
        best = np.argsort(metrics)[::-1][1:n+1]
        best_metrics = metrics[best]
        return best, best_metrics
    
    
    def analogy(self, pos, neg, n=10):
        """
        Analogy similarity.

        Parameters
        ----------
        pos : list
        neg : list

        Returns
        -------
        2 numpy.array:
            1. position in self.vocab
            2. cosine similarity

        Example
        -------
            `king - man + woman = queen` will be:
            `pos=['king', 'woman'], neg=['man']`
        """
        exclude = pos + neg
        pos = [(word, 1.0) for word in pos]
        neg = [(word, -1.0) for word in neg]

        mean = []
        for word, direction in pos + neg:
            mean.append(direction * self[word])
        mean = np.array(mean).mean(axis=0)

        metrics = np.dot(self.vectors, mean)
        best = metrics.argsort()[::-1][:n + len(exclude)]

        exclude_idx = [np.where(best == self.ix(word)) for word in exclude if
                       self.ix(word) in best]
        new_best = np.delete(best, exclude_idx)
        best_metrics = metrics[new_best]
        return new_best[:n], best_metrics[:n]

    def generate_response(self, indexes, metrics, clusters=True):
        '''
        Generates a pure python (no numpy) response based on numpy arrays
        returned by `self.cosine` and `self.analogy`
        '''
        if self.clusters and clusters:
            return np.rec.fromarrays((self.vocab[indexes], metrics,
                                     self.clusters.clusters[indexes]),
                                     names=('word', 'metric', 'cluster'))
        else:
            return np.rec.fromarrays((self.vocab[indexes], metrics),
                                     names=('word', 'metric'))

    def to_mmap(self, fname):
        if not joblib:
            raise Exception("sklearn is needed to save as mmap")

        joblib.dump(self, fname)
    
    def set_hidden_words(self):
        if self.train:
            if not self.train['hs']:
                self.hidden_words = []
                for key, items in self.train['syn1'].iteritems():
                    if "_*" not in key: 
                        self.hidden_words.append(key)
            
            self.hword_len = len(self.hidden_words)

    @staticmethod
    def read_hidden_layer(cls,fname, kind='text'):
        if kind=='text':
            return cls.read_hidden_layer_text(fname)


    @staticmethod
    def read_hidden_layer_bin(fname):
        """
        read hidden layers - binary
        """
        fsyn1 = fname + ".syn1"
        if not os.path.isfile(fsyn1):
            return None
        
        model = {}
        with open(fsyn1, 'r') as f:
            header = f.readline()
            tokens = header.strip().split()
            for idx in xrange(0, len(tokens), 2):
                model[tokens[idx]] = int(tokens[idx+1])

            syn1 = {}
            for i, line in enumerate(f):
                idx = line.index(" ")
                key = line[0:idx]
                val = line[idx+1:]
                vector = np.fromstring(val, dtype=np.float)
                syn1[key] = vector

            model['syn1'] = syn1
        
        if model['hs']:
            fvocab = fname + ".vocab"
            if not os.path.isfile(fvocab):
                return None
            vocab = {}
            with open(fvocab, 'r') as f:
                for line in f:
                    tokens  = line.strip().split(' ')
                    ventry = {}
                    ventriy['word'] = tokens[0]
                    ventry['codelen'] = int(tokens[1])
                    point = []
                    code = []
                    idx = line.index(" ")
                    line = line[idx+1:]
                    idx = line.index(" ")

                    vector = np.fromstring(line[idx+1:], dtype=np.int)

                    for idx in xrange(ventry['codelen']):
                        point.append(str(vector[idx]))
                    for idx in xrange(ventry['codelen']):
                        code.append(int(vector[idx + ventry['codelen']]))

                    ventry['point'] = point
                    ventry['code'] = code
                    vocab[ventry['word']] = ventry

            model['vocab'] = vocab
        else:
            model['vocab'] = None
        
        return model

    @staticmethod
    def read_hidden_layer_text(fname):
        """
        Read hidden layers - text
        """
        fsyn1 = fname + ".syn1"
        if not os.path.isfile(fsyn1):
            return None

        model = {}
        with open(fsyn1, 'r') as f:
            header = f.readline()
            tokens = header.strip().split()
            for idx in xrange(0, len(tokens), 2):
                model[tokens[idx]] = int(tokens[idx+1])
            
            syn1 = {}
            for i, line in enumerate(f):
                tokens = line.strip().split(' ')
                key =  tokens[0]
                vector = np.array(tokens[1:], dtype=np.float)
                syn1[key] = vector

            model['syn1'] = syn1
            
        if model['hs']:
            fvocab = fname + ".vocab"
            if not os.path.isfile(fvocab):
                return None
            vocab = {}
            with open(fvocab, 'r') as f:
                for line in f:
                    tokens  = line.strip().split(' ')
                    ventry = {}
                    ventry['word'] = tokens[0]
                    ventry['codelen'] = int(tokens[1])
                    point = []
                    code = []
                    for idx in xrange(ventry['codelen']):
                        point.append(tokens[2+idx])
                        code.append(int(tokens[2+ventry['codelen']+idx]))

                    ventry['point'] = point
                    ventry['code'] = code
                    vocab[ventry['word']] = ventry

            model['vocab'] = vocab
        else:
            model['vocab'] = None

        return model
    
    def save_comp_model(self, fname):
        #text mode
        tmp_errors = []
        floatStruct = struct.Struct('f')
        with open(fname, 'wb') as fo:
            fo.write("%d %d\n" % (self.hword_len, len(self.vectors[0])))
            for a in xrange(self.hword_len):
                token = self.hidden_words[a]
                if a <= 2:
                    val = self.get_vector(token)
                    data = floatStruct.pack(val[0])
                
                if token in self and token:
                    #for ch in token: 
                    fo.write(token + " ")
                    for val in self.get_vector(token):
                        data = floatStruct.pack(val)
                        fo.write(data)
                    fo.write("\n")
                else:
                    tmp_errors.append(token)
        
        if self.train:
            with open(fname + ".syn1", 'wb') as fsyn1:
                fsyn1.write("cbow %d " % self.train['cbow'])
                fsyn1.write("hs %d " % self.train['hs'])
                fsyn1.write("neg %d " % self.train['neg'])
                fsyn1.write("window %d " % self.train['window'])
                fsyn1.write("layer1_size %d " % self.train['layer1_size'])
                fsyn1.write("epochs %d " % self.train['epochs'])
                if self.train['hs'] == 0:
                    fsyn1.write("syn1_size %d\n" % self.hword_len)
                    for a in xrange(self.hword_len):
                        fsyn1.write("%s " % self.hidden_words[a])
                        for val in self.train['syn1'][self.hidden_words[a]]:
                            fsyn1.write("%f " % val)
                        fsyn1.write("\n")
                    
    
    @classmethod
    def from_binary(cls, fname, vocabUnicodeSize=86, desired_vocab=None):
        """
        Create a WordVectors class based on a word2vec binary file

        Parameters
        ----------
        fname : path to file
        vocabUnicodeSize: the maximum string length (78, by default)
        desired_vocab: if set, this will ignore any word and vector that
                       doesn't fall inside desired_vocab.

        Returns
        -------
        WordVectors instance
        """
        train = cls.read_hidden_layer_text(fname)
        with open(fname, 'rb') as fin:
            header = fin.readline()
            vocab_size, vector_size = list(map(int, header.split()))

            vocab = np.empty(vocab_size, dtype='<U%s' % vocabUnicodeSize)
            vectors = np.empty((vocab_size, vector_size), dtype=np.float)
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for i in range(vocab_size):
                # read word
                word = ''
                while True:
                    ch = fin.read(1).decode('ISO-8859-1')
                    if ch == ' ':
                        break
                    word += ch
                include = desired_vocab is None or word in desired_vocab
                if include:
                    vocab[i] = word

                # read vector
                vector = np.fromstring(fin.read(binary_len), dtype=np.float32)
                if include:
                    vectors[i] = unitvec(vector)
                fin.read(1)  # newline

            if desired_vocab is not None:
                vectors = vectors[vocab != '', :]
                vocab = vocab[vocab != '']
        return cls(vocab=vocab, vectors=vectors, train=train)

    @classmethod
    def from_text(cls, fname, vocabUnicodeSize=78, desired_vocab=None):
        """
        Create a WordVectors class based on a word2vec text file

        Parameters
        ----------
        fname : path to file
        vocabUnicodeSize: the maximum string length (78, by default)
        desired_vocab: if set, this will ignore any word and vector that
                       doesn't fall inside desired_vocab.

        Returns
        -------
        WordVectors instance
        """
        with open(fname, 'rb') as fin:
            header = fin.readline()
            vocab_size, vector_size = list(map(int, header.split()))

            vocab = np.empty(vocab_size, dtype='<U%s' % vocabUnicodeSize)
            vectors = np.empty((vocab_size, vector_size), dtype=np.float)
            for i, line in enumerate(fin):
                line = line.decode('ISO-8859-1').strip()
                parts = line.split(' ')
                word = parts[0]
                include = desired_vocab is None or word in desired_vocab
                if include:
                    vector = np.array(parts[1:], dtype=np.float)
                    vocab[i] = word
                    vectors[i] = unitvec(vector)

            if desired_vocab is not None:
                vectors = vectors[vocab != '', :]
                vocab = vocab[vocab != '']
        
        train = cls.read_hidden_layer(fname)
        return cls(vocab=vocab, vectors=vectors, train=train)
    


    @classmethod
    def from_mmap(cls, fname):
        """
        Create a WordVectors class from a memory map

        Parameters
        ----------
        fname : path to file

        Returns
        -------
        WordVectors instance
        """
        memmaped = joblib.load(fname, mmap_mode='r+')
        train = cls.read_hidden_layer(fname)
        return cls(vocab=memmaped.vocab, vectors=memmaped.vectors, train=train)
