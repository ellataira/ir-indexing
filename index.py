import collections
import os

import Utils
util = Utils()
FILEPATH = "/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/"

class Indexer:

    def __init__(self, parsed_doc_term_dict_filename, is_stemmed):
        self.parsed_doc_term_dict_filename = parsed_doc_term_dict_filename
        # inverted index will consist of term : docid : [posns]
        self.inverted_index = {}
        self.index_stats = {}
        self.doc_lens = {}
        self.is_stemmed = is_stemmed
        self.sum_terms = 0

    # given a single document, adds to inverted index
    # catalog (term, start offset, length)
    # and inverted file
    def index_doc(self, tokens, partial_dict):
        for term, docid, posn in tokens:  # tpl = (token, self.doc_id, posn)
            self.sum_terms += 1
            # if term hasn't been added to index yet
            if term not in partial_dict:
                partial_dict[term] = {}
            # if doc is not in term's dict yet
            if docid not in partial_dict:
                partial_dict[term][docid] = []
            # add posn to index -- term : docid : [posns]
            partial_dict[term][docid].append(posn)

        return partial_dict

    # indexes 1000 documents at a time
    # saves each partial index when 1000 docs have been indexed
    def generate_partial_indices(self):
        docs_per_partial = 1000
        doc_count = 0
        partial_dict = {}
        partial_dict_num = 0
        token_dict = util.read_pickle(self.parsed_doc_term_dict_filename)
        for docid, tokens in token_dict.items():
            print(docid)
            self.doc_lens[docid] = len(tokens)
            # if exceed 1000 documents indexed, save old index and restart new partial index
            if doc_count >= docs_per_partial:
                self.save_partial_dict(partial_dict, partial_dict_num)
                partial_dict = {}
                doc_count = 0
                partial_dict_num += 1
            partial_dict = self.index_doc(tokens, partial_dict)
            doc_count += 1

        # after creating all partial indices, save term vector and doc len pickles
        util.save_dict(FILEPATH + "doc_lengths_" + self.is_stemmed + ".pkl", self.doc_lens)
        print("saved doc length dictionary")

    # saves a partial index (txt) and corresponding catalog (pkl)
    def save_partial_dict(self, partial_dict, partial_dict_num):
        # sort partial dict into alphabetical order by term (key)
        partial_dict = sorted(partial_dict.items())

        # index will be stored as a pkl dictionary
        partial_index = FILEPATH + self.is_stemmed + "/index" +\
                        "/partial_index_" + self.is_stemmed + str(partial_dict_num) + ".txt"

        # partial catalog will be stored as a txt file
        partial_catalog_filename = FILEPATH  + self.is_stemmed + "/catalog" + "/partial_catalog_" \
                          + self.is_stemmed + str(partial_dict_num) + ".pkl"

        partial_catalog_dict = {}

        w_partial_index = open(partial_index, "w")

        offset = 0
        for term, docs in partial_dict:
            index_entry = ""
            for docid, postings in docs.items():
                index_entry += str(docid)
                s= " ".join(str(postings))
                index_entry += s.replace("[", "").replace("]", "")
                index_entry += "\n"
            w_partial_index.write(index_entry)
            partial_catalog_dict[term] = (offset, len(index_entry)) # partial catalog pkl = dict[term] = (offset, length)
            offset += len(index_entry)

        w_partial_index.close()
        print("saved partial index and catalog number: " + str(partial_dict_num))
        util.save_dict(partial_catalog_filename, partial_catalog_dict)

    # converts catalog to dictionary given the catalog pkl path
    def catalog_to_dict(self, catalog_path):
        catalog_as_dict = {}
        with open(catalog_path) as catalog:
            for line in catalog:
                tokens = line.split(" ")
                term, offset, length = tokens[0], tokens[1], tokens[2].strip()
                catalog_as_dict[term] = (int(offset), int(length))

        return collections.OrderedDict(catalog_as_dict)

    # catalog1 will always be the shorter list
    # merges two index-catalog pairs
    def merge_indices(self, catalog1, catalog2, index1, index2, merge_count):
        new_cat_path = FILEPATH + self.is_stemmed + "/merged/" + "new_cat_merge_no_" + str(merge_count) + ".pkl"
        new_cat_dict = {}

        new_index_path = FILEPATH + self.is_stemmed + "/merged/" + "new_index_no_" + str(merge_count) + ".txt"
        new_index = open(new_index_path, 'w')

        left_cat = util.read_pickle(catalog1)
        right_cat = util.read_pickle(catalog2)

        merged_offset = 0

        # merge until shorter catalog is empty
        for term, data in left_cat.items():

            # read left index
            l_offset, l_length = left_cat[term]
            open_left_index = open(index1, "r")
            open_left_index.seek(l_offset)
            l_to_write = open_left_index.read(l_length)
            open_left_index.close()

            # if term is not in right catalog, then write info from left catalog to master
            if term not in right_cat:
                to_write = l_to_write
            # if term is in right catalog, read right index, combine with left index, and write to master
            elif term in right_cat:
                # find data from right
                r_offset, r_length = right_cat[term]
                open_right_index = open(index2, "r")
                open_right_index.seek(r_offset)
                r_to_write = open_right_index.read(r_length)
                open_right_index.close()

                # delete term from right catalog so when left catalog is empty, we can just append remaining unseen
                # terms from right to end of master catalog
                del right_cat[term]

                to_write = l_to_write + r_to_write

            new_cat_dict[term] = (merged_offset, len(to_write))
            new_index.write(to_write)
            merged_offset += len(to_write)

        # now only adding the unseen terms from right catalog
        for term, data in right_cat.items():
            # deal with remaining terms in right cat (those not in left cat)
            # add to new cat
            r_offset, r_length = right_cat[term]
            open_right_index = open(index2, "r")
            open_right_index.seek(r_offset)
            r_to_write = open_right_index.read(r_length)
            open_right_index.close()

            new_cat_dict[term] = (merged_offset, len(r_to_write))
            new_index.write(r_to_write)
            merged_offset += len(r_to_write)

        new_index.close()

        cat_keys = list(new_cat_dict.keys())
        cat_keys.sort()
        sorted_cat_dict = {i: new_cat_dict[i] for i in cat_keys}
        util.save_dict(new_cat_path, sorted_cat_dict)
        return new_cat_path, new_index_path

    # merges all partial indexes and catalogs
    def merge(self):

        catalogs = os.listdir(FILEPATH + self.is_stemmed + "/catalog")

        partial_index_base_path = FILEPATH + self.is_stemmed + "/index" + \
                        "/partial_index_" + self.is_stemmed

        # partial catalog will be stored as a txt file
        partial_catalog_base_path = FILEPATH + self.is_stemmed + "/catalog" + "/partial_catalog_" \
                                   + self.is_stemmed

        master_cat = partial_catalog_base_path + '0.pkl'
        master_index = partial_index_base_path + '0.txt'

        for i in range(0, len(catalogs)):
            print(i)
            lc = partial_catalog_base_path + str(i) + '.pkl'
            print(lc)
            li = partial_index_base_path + str(i) + '.txt'
            print(li)

            master_cat, master_index = self.merge_indices(catalog1=lc, catalog2=master_cat,
                                                          index1=li, index2=master_index, merge_count=i)


if __name__ == '__main__':
    indexer = Indexer("/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/stemmed/stemmed_document_dict.pkl", "stemmed")
    indexer.generate_partial_indices()
    indexer.merge()
    indexer = Indexer("/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/unstemmed/unstemmed_document_dict.pkl", "unstemmed")
    indexer.generate_partial_indices()
    indexer.merge()
