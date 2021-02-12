import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from string import punctuation
from collections import OrderedDict
from nltk.probability import FreqDist

from fecir.bm25 import *
from fecir.dci_closed import dci_closed, extract_itemsets

'''
    Retriever from 
    `Fast and Effective Cluster-based Information Retrieval 
        Using Frequent Closed Itemsets` 
'''
class Retriever:
    def __init__(self, docs, random_state = 0):
        self.docs = docs
        self.length = len(docs)
        self.random_state = random_state
        self.clusters = list()
        return

    '''
        Runs preprocessing step
    '''
    def build(self, k, min_sup = 0.5, stopwords_lang = "portuguese", domain_stopwords = None):
        self.k = k

        # Generating clean documents (no stopwords)
        self.vsm = self._vsm_build(stopwords_lang, domain_stopwords)
        
        # Applying K-Means to the VSM
        self.doc_labels = self._cluster_build(k)         
        
        # Building clusters with documents
        for i in range(k):
            new_cluster = Cluster(label=i)
            self.clusters.append(new_cluster)
        for i in range(self.length):
            self.clusters[self.doc_labels[i]].addDoc(self.docs[i])

        # Cleaning and tokenizing words in clusters and building itemsets
        # with DCI-Closed
        for i in range(k):
            self.clusters[i].DCIClosed(self.stopwords, min_sup, lang = stopwords_lang)
        return
    
    '''
        Looks for documents that match a given query
         (Information Retrieval step)
    '''
    def search(self, query):
        # Generating word frequency dictionary
        word_dict = self._tokenize_query(query)

        # Matching words with itemsets
        matching = {}
        for i in range(self.k):
            value = 0
            for term in word_dict.keys():
                value += self._search_term(term, self.clusters[i].itemsets)
            matching['Assunto, F' + str(i + 1)] = value

        # Ranking clusters based on scores 
        scores = list(matching.values())
        ranked_cluster = [x for _, x in sorted(zip(scores,self.clusters), key=lambda pair: pair[0])]
        ranked_cluster.reverse()
        return ranked_cluster

    def _tokenize_query(self, query):
        terms = word_tokenize(query.lower())
        terms = [word for word in terms if word not in self.stopwords]
        
        # Build word frequency dict
        term_frequency = OrderedDict(sorted(FreqDist(terms).items(), key=lambda x: x[1],reverse=True))
        return term_frequency

    def _search_term(self, term, itemset):
        if (len(itemset)) != 0:
            for item in itemset:
                if term in item:
                    return 1
        return 0

    '''
        Generates Vector Space Model using TF-IDF
        and ignoring selected stopwords
    '''
    def _vsm_build(self, stopwords_lang = "portuguese", domain_stopwords = None):
        self.stopwords = nltk.corpus.stopwords.words(stopwords_lang)
         
         # Adding domain specific stopwords
        if (domain_stopwords != None):
            self.stopwords.extend(domain_stopwords)
        self.stopwords.extend([",", ".", "(", ")"])

        # Generating Vecotor Space Model
        vectorizer = TfidfVectorizer(stop_words=self.stopwords)
        vsm = vectorizer.fit_transform(self.docs)

        return vsm

    def _cluster_build(self, k):
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, max_iter=1)
        clusters = kmeans.fit(self.vsm)
        return clusters.labels_



'''
    Cluster
'''
class Cluster:
    def __init__(self,label):
        self.label = label
        self.elements = list()
        self.size = 0
        return
    
    def addDoc(self, doc):
        #print(f"=====\nAdding to cluster {self.label}>> {doc}\n=====\n")
        self.size += 1
        self.elements.append(doc)
        return

    def DCIClosed(self, stopwords, min_sup = 0.5, lang = "portuguese"):
        self._bagOfWords_build(stopwords, lang)
        
        closed_set, pre_set, post_set = extract_itemsets(self.words, min_sup)
        self.itemsets = dci_closed(closed_set, pre_set, post_set, self.words, min_sup)        
        return

    def _bagOfWords_build(self, stopwords, lang = "portuguese"):
        clean_words = list()
        for doc in self.elements:
            doc = doc.replace("\r\n", " ")
            doc_words = [word.lower() for word in word_tokenize(doc, language=lang) if word not in stopwords]
            clean_words.append(doc_words)
        self.words = clean_words
        #print(clean_words)
        return