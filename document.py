import string
from nltk import regexp_tokenize, PorterStemmer


class Document:

    # tokenize a document to convert it to a sequence of (term_id, doc_id, position)
    # only tokenizes, does not stem or remove stop words TODO: should i remove stop words now? they arent in either index type
    # @param remove_stopwords = True or False
    def __init__(self, docid, text, stem):
        self.doc_id = docid
        self.doc_length = 0
        self.stops = self.read_stop_words('/Users/ellataira/Desktop/is4200/homework-2-ellataira/Code/stoplist.txt')
        self.tokens = self.tokenize(text, stem)
        self.doc_length = len(self.tokens)


    # tokenize() will take in the text found by preprocess.parse() in <TEXT> </TEXT>
    def tokenize(self, text, stem):
        result = []
        stemmer = PorterStemmer()
        # generate tokens
        tokens = lambda x: x, regexp_tokenize(text, pattern=r"\w+(?:\.?\w)*")

        # convert tokens to (term_id, doc_id, position)
        for posn, token in enumerate(tokens[1]):
            token = token.lower()
            if token in self.stops:
                continue
            else:
                #  if creating a stemmed index , then stem token
                if stem:
                    token = stemmer.stem(token)
                tpl = (token, self.doc_id, posn)
                result.append(tpl)

        self.doc_length = len(result)
        return result

    # converts stop words file to array of terms
    def read_stop_words(self, filename):
        lines = []
        with open(filename, 'r') as f:
            for line in f:
                lines.append(line.strip())
        return lines
