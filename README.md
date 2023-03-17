# HW2: Indexing

The objective of this project is to implement an indexer, used in place of the Elasticsearch indexer. Then, generate the index from a set of documents and search the index to return documents relevant to a list of queries. 

This project is a continuation of [ir-retrieval-models](https://github.com/ellataira/ir-retrieval-models)
. 

## Task1: Tokenizing
The file, `document.py`, contains all methods used to tokenize and given document. The method `tokenize()` uses 
`regexp_tokenize()` and regular expressions to retrieve tokens and their positions. Stop words are removed from the returned list 
of tokens. When creating a stemmed index, each token is stemmed using PorterStemmer. 

## Task2: Indexing
The file, `parser.py` is adapted from homework 1, but calls the Document class and its tokenizer. 'Parser' saves a dictionary that maps 
each document number to its list of tokens (an un-inverted index). These dictionaries, one for unstemmed tokens and one for stemmed,
are saved as pkl files. 

To save space, each document's doc ID has been mapped to a shorted version. These shorted doc IDs are used for the remainder of the program. 
The mappings between shorted doc ID and its true doc ID are stored in `docid_dict.pkl`. 

The document lengths for each document in the collection are stored as a dictionary `docid : length` in `doc_lengths_stemmed.pkl`
and `doc_lengths_unstemmed.pkl`. 

After generating the un-inverted indexes, `index.py` converts each dictionary to an inverted index in the format 
`term : docid : [posns]`. For every 1000 documents read, a partial index and catalog are generated and saved. The partial index
contains the doc ID and a term's positions. The catalog contains a term, the offset, and the length of data, referencing the partial index. 
The partial index is saved as a txt file, and the catalog is saved as a pkl. 

After generating partial indexes and catalogs for every document in the collection, they are merged together to create one 
single master catalog and index. `merge()` merges two corresponding index-catalog pairs at a time. The final merged catalog is 
`new_cat_merge_no_83.pkl` and the final merged index is `new_index_no_83.txt`. 

Two merged index-catalog pairs are created, one where tokens are stemmed and another where tokens are unstemmed.  

| Unstemmed Index Size | Stemmed Index Size |
|----------------------|--------------------|
| 173.8 MB             | 163.6 MB           |

## Task3: Searching

The file, `query_execution.py`, has been modified from Homework 1 to use the new index-catalog system. Helper methods that retrieve 
values such as term frequency and document length have been adjusted to read from stored data. The three ranking models 
are Okapi-TF, Okapi-BM25, and Unigram LM with Laplace smoothing score. 

### Uninterpolated mean average precision
|                   | **Okapi TF** | **Okapi BM25** | **La Place** | 
|-------------------|-------------|--------|--------------| 
| **HW1**           | 0.2553      | 0.2957 | 0.2477       |
| **HW2 Stemmed**   | 0.1790      | 0.1906 | 0.1910       | 
| **HW2 Unstemmed** | 0.1614      | 0.1656 | 0.1717       | 

