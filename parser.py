import os
import re
import pickle

from Code.Utils import Utils
from document import Document

"""
    regex syntax: 
    '.' operator == matches any character 
    '*' operator == repeat preceding operator (i.e. any char) 0+ times
    '?' operator == repeat preceding char 0 or 1 times 
"""
DOC_REGEX = re.compile("<DOC>.*?</DOC>", re.DOTALL) ## DOTALL allows '.' to equal any character, including \n
DOCNO_REGEX = re.compile("<DOCNO>.*?</DOCNO>")
TEXT_REGEX = re.compile("<TEXT>.*?</TEXT>", re.DOTALL)

class Parser:
    def __init__(self, stem):
        # for use in model calculations which require vocab_size
        self.vocab = []
        self.documents = {}
        self.docids = {}
        self.stem = stem
        # self.open_dir("/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/test_collection")
        self.open_dir("/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/ap89_collection")


    """opens file collection and delegates to parse individual files """
    def open_dir(self, data):
        entries = os.listdir(data)
        id = 1

        print(id)
        # for every 'ap....' file in the opened directory, parse it for documents
        for entry in entries:
            if 'ap' in entry: ## excludes the readme file
                filepath = data + "/" + entry
                id = self.parse(filepath, id)
                print("parsed: "+ filepath + "\n")


    """parses an individual file from the collection for documents / info """
    def parse(self, filepath, id):

        with open(filepath, encoding="ISO-8859-1") as opened:

            read_opened = opened.read()
            found_docs = re.findall(DOC_REGEX, read_opened)

            print(id)
            for doc in found_docs:
                found_doc = re.search(DOCNO_REGEX, doc)
                docno = re.sub("(<DOCNO> )|( </DOCNO>)", "", found_doc[0])

                found_text = re.search(TEXT_REGEX, doc)
                text = re.sub("(<TEXT>\n)|(\n</TEXT>)", "", found_text[0])
                text = re.sub("\n", " ", text)

                tokens = Document(id, text, stem=self.stem).tokens

                self.docids[id] = docno
                self.documents[id] = tokens
                id += 1

            return id


if __name__ == '__main__':
    stemmed = Parser(stem=True)
    util = Utils()
    util.save_dict('/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/docid_dict.pkl', stemmed.docids)
    util.save_dict('/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/stemmed/stemmed_document_dict.pkl', stemmed.documents)
    unstemmed = Parser(stem=False)
    util.save_dict('/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/unstemmed/unstemmed_document_dict.pkl', unstemmed.documents)
