import math
import os
import pickle

from nltk import regexp_tokenize
from nltk.stem.porter import *
from Utils import Utils
util = Utils()

from elasticsearch7 import Elasticsearch

es = Elasticsearch("http://localhost:9200", timeout=900000000)
AP89_INDEX = 'ap89_index'
q_data = "/Users/ellataira/Desktop/is4200/homework-1-ellataira/IR_data/AP_DATA/new_queries.txt"
# q_data = "/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/test_queries.txt"

VOCAB_SIZE = 288141
TOTAL_DOCS = 84678
SIZE = 1000

infile = open('/Users/ellataira/Desktop/is4200/homework-1-ellataira/IR_data/AP_DATA/doc_lens_dict.pkl', 'rb')
DOC_LENS = pickle.load(infile)
infile.close()

######################## PROCESS QUERIES ##################################################################

# modify input queries through stemming, shifting to lowercase, and removing stop words
# stores the modified queries in a dictionary as key-value : (qID, query)
def query_analyzer(query, stem):
    result = []
    stemmer = PorterStemmer()
    # generate tokens
    tokens = lambda x: x, regexp_tokenize(query, pattern=r"\w+(?:\.?\w)*")

    stopwords = []
    with open('/Users/ellataira/Desktop/is4200/homework-2-ellataira/Code/stoplist.txt', 'r') as f:
        for line in f:
            stopwords.append(line.strip())

    # convert tokens to (term_id, doc_id, position)
    for posn, token in enumerate(tokens[1]):
        token = token.lower()
        if token in stopwords:
            continue
        else:
            if stem:
                token = stemmer.stem(token)
            result.append(token)

    return result


# anaylzes .txt files of queries and stores them as key-value: qID, query terms (modified)
def process_all_queries(query_file, to_stem):
    with open(q_data, encoding="ISO-8859-1") as opened:
        lines = opened.readlines()

        query_dict = {}

        for line in lines:
            sects = line.split(".")
            q_id = sects[0].strip()
            mod_q = query_analyzer(sects[1].strip(), to_stem)
            query_dict[q_id] = mod_q

    opened.close()
    return query_dict


######################## HELPER FUNCTIONS / "GET"-[] ##################################################################

# returns term frequency of given term in a given query
def get_word_in_query_frequency(term, query):
    count = 0
    for s in query:
        if s == term:
            count += 1
    return count

def calc_avg_doc_length(doc_lengths_dict):
    s = sum(doc_lengths_dict.values())
    q = len(doc_lengths_dict)
    return s / q

# returns the length (number of words) in a specified document
def get_doc_length(d_id, doc_lengths_dict):
    return doc_lengths_dict[int(d_id)]

# find term frequency in all documents in corpus
def get_doc_frequency_of_word(relevant_doc_dict):
    s=0
    for d in relevant_doc_dict.values():
        s += len(d)
    return s

# opens an index .txt file and returns text from given offset of given length
def read_index(index_path, offset, length):
    open_index = open(index_path, "r")
    open_index.seek(offset)
    data = open_index.read(length)
    open_index.close()
    return data

# given a section of the index document (as text), generates partial dictionary of docid -> [posns]
def generate_dict_from_index(relevant_docs):
    ret_dict = {}
    lines = relevant_docs.split("\n")
    for l in lines:
        l.strip()
        split_on_space = l.split(" ")
        docid = split_on_space[0]
        del split_on_space[0]
        if docid != "":
            ret_dict[docid] = split_on_space[:-1]
    return ret_dict

# returns long version of docid given shorted hash
def get_real_docno(short_docno, docno_dict):
    return docno_dict[int(short_docno)]

def get_all_rel_docs(query_arr, catalog, index_path):
    master_dict = {}
    for term in query_arr:
        get_term = catalog.get(term, 0)
        if get_term:
            offset, length = get_term
            relevant_docs = read_index(index_path, offset, length)
            relevant_doc_dict = generate_dict_from_index(relevant_docs)
            master_dict = master_dict | relevant_doc_dict
    return master_dict


######################### OUTPUT  RESULTS / SORT #########################################################

# sorts documents in descending order, so the doc with the highest score is first (most relevant)
# and truncates at k docs
def sort_descending(relevant_docs, k):
    sorted_docs = sorted(relevant_docs.items(), key=lambda item: float(item[1]), reverse=True)
    del sorted_docs[k:]
    return sorted_docs

# saves a list of scored docs to a .txt file
# @param 2-d dictionary of scored documents [query][documents]
def save_to_file(relevant_docs, filename):
    f = '/Users/ellataira/Desktop/is4200/homework-2-ellataira/Results/' + filename + '.txt'
    k = SIZE  # want to save the top 1000 files

    if os.path.exists(f):
        os.remove(f)

    with open(f, 'w') as f:
        for query_id, results_dict in relevant_docs.items():
            sorted_dict = sort_descending(results_dict, k)
            count = 1
            for d_id, score in sorted_dict:
                f.write(str(query_id) + ' Q0 ' + str(d_id) + ' ' + str(count) + ' ' + str(score) + ' Exp\n')
                count += 1

    f.close()

######################## OKAPI TF ##################################################################

# calculates okapi tf score of a single document
def okapi_tf(tf_wd, dl, adl):
    score = tf_wd / (tf_wd + 0.5 + (1.5 * (dl / adl)))
    return score

######################## Okapi BM25 ##################################################################

# calculates the okapi bm25 score of a single document
def okapi_bm25(tf_wq, tf_wd, df_w, adl, dl, d):
    b = 0.75
    k1 = 1.2
    k2 = 100  # k2 param is typically 0 <= k2 <= 1000
    log = math.log((d + 0.5) / (df_w + 0.5))
    a = (tf_wd + k1 * tf_wd) / (tf_wd + k1 * ((1 - b) + b * (dl / adl)))
    b = (tf_wq + k2 * tf_wq) / (tf_wq + k2)
    return log * a * b

######################## Unigram LM with Laplace smoothing ##################################################

# calculates Unigram LM with Laplace smoothing score of a single document
def uni_lm_laplace(tf_wd, dl, v):
    p_laplace = (tf_wd + 1) / (dl + v)
    return math.log(p_laplace)

##########################################################################################################

# method that calls all vector ranking models
# will increment a document score only if the term is in the current document
def Vector_Prob_Models(queries, catalog, index_path, doc_lengths, avg_dl, docid_dict):
    # initialize scores dictionaries
    okapi_scores = {}
    okapi_bm25_scores = {}

    # for each query,
    for id, query in queries.items():
        q_id = id
        print("q_id: " + str(q_id))
        print("query: " + str(query))

        # for each query, instaniate its index in 2-d solution array
        okapi_scores[q_id] = {}
        okapi_bm25_scores[q_id] = {}

        d = len(doc_lengths)

        # for each term in query,
        for term in query:
            get_term = catalog.get(term, 0)
            if get_term:
                offset, length = get_term
                relevant_docs = read_index(index_path, offset, length)
                relevant_doc_dict = generate_dict_from_index(relevant_docs)

                # only calculate score if the term is in the document
                for docid, posns in relevant_doc_dict.items():
                    tf_wq = get_word_in_query_frequency(term, query)
                    tf_wd = len(posns)
                    dl = get_doc_length(docid, doc_lengths)
                    adl = avg_dl
                    df_w = get_doc_frequency_of_word(relevant_doc_dict)

                    real_did = get_real_docno(docid, docid_dict)

                    # okapi-tf
                    okapi_score = okapi_tf(tf_wd, dl, adl)
                    try:
                        okapi_scores[q_id][real_did] += okapi_score
                    except (KeyError):
                        okapi_scores[q_id][real_did] = okapi_score

                    # Okapi BM25
                    okapi_bm25_score = okapi_bm25(tf_wq, tf_wd, df_w, adl, dl, d)
                    try:
                        okapi_bm25_scores[q_id][real_did] += okapi_bm25_score
                    except (KeyError):
                        okapi_bm25_scores[q_id][real_did] = okapi_bm25_score

    return okapi_scores, okapi_bm25_scores

# ranks with Unigram LM with Laplace smoothing model
def Unigram_Models(queries, catalog, index_path, doc_lengths, docid_dict):
    # initialize scores dictionaries
    laplace_scores = {}

    # for each query,
    for id, query in queries.items():
        q_id = id
        print("q_id: " + str(q_id))
        print("query: " + str(query))

        # for each query, instaniate its index in 2-d solution array
        laplace_scores[q_id] = {}

        v = len(catalog)

        all_rel_docs = get_all_rel_docs(query, catalog, index_path)

        # for each term in query,
        for term in query:
            get_term = catalog.get(term, 0)
            if get_term:
                offset, length = get_term
                relevant_docs = read_index(index_path, offset, length)
                term_relevant_doc_dict = generate_dict_from_index(relevant_docs)

                # for all docs relevant to any/all terms in query
                for docid, posns in all_rel_docs.items():
                    # if term is in specific doc
                    if docid in term_relevant_doc_dict.keys():
                        # score document regardless of if the term appears in it
                        doc_posns = term_relevant_doc_dict[docid]
                        tf_wd = len(doc_posns)  # tf of word in document is equal to the number of occurrences (i.e. positions)
                        dl = get_doc_length(docid, doc_lengths)
                    # if term is not in specific doc
                    else:
                        tf_wd = 0
                        dl = 100

                    # if the term is not present in document, then tf_wd = 0
                    uni_lm_laplace_score = uni_lm_laplace(tf_wd, dl, v)

                    real_did = get_real_docno(docid, docid_dict)

                    try:
                        laplace_scores[q_id][real_did] += uni_lm_laplace_score
                    except (KeyError):
                        laplace_scores[q_id][real_did] = uni_lm_laplace_score

    return laplace_scores


# runs all ranking models and saves their outputs to txt files
# processes queries, searches for relevant documents, and then scores those documents using the
# different ranking models
def run_all_models(index_path, catalog_path, to_stem):
    # process, stem, remove stop words from queries
    queries = process_all_queries(q_data, to_stem)
    print(queries)

    catalog = util.read_pickle(catalog_path)

    stem_term = "stemmed" if to_stem else "unstemmed"

    doc_lengths = util.read_pickle("/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/doc_lengths_"
                                   + stem_term + ".pkl")

    avg_dl = calc_avg_doc_length(doc_lengths)

    docid_dict = util.read_pickle("/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/docid_dict.pkl")

    # vector / prob models
    okapi_scores, okapi_bm25_scores = Vector_Prob_Models(queries, catalog, index_path, doc_lengths, avg_dl, docid_dict)

    save_to_file(okapi_scores, "okapi_tf_" + stem_term)
    print("saved okapi scores")

    save_to_file(okapi_bm25_scores, "okapi_bm25_" + stem_term)
    print("saved okapi bm25 scores")

    # language models
    uni_lm_laplace_scores = Unigram_Models(queries, catalog, index_path, doc_lengths, docid_dict)

    save_to_file(uni_lm_laplace_scores, "uni_lm_laplace_" + stem_term)
    print("saved uni lm laplace scores")

    print("complete!")


if __name__ == '__main__':
    # stemmed_index = "/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/stemmed/merged/new_index_no_83.txt"
    # stemmed_cat = "/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/stemmed/merged/new_cat_merge_no_83.pkl"
    # run_all_models(stemmed_index, stemmed_cat, True)
    unstemmed_index = "/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/unstemmed/merged/new_index_no_83.txt"
    unstemmed_cat = "/Users/ellataira/Desktop/is4200/homework-2-ellataira/data/unstemmed/merged/new_cat_merge_no_83.pkl"
    run_all_models(unstemmed_index, unstemmed_cat, False)

