import collections
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import pandas as pd

def cluster_sentences(sentences, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.99,
                                        min_df=0.1,
                                        lowercase=True)
        #builds a tf-idf matrix for the sentences
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)


if __name__ == "__main__":
    data = pd.read_csv('../data/formated.csv')
    sentences = data['previousStep'].values
    print(sentences)
    nclusters= 3
    clusters = cluster_sentences(sentences, nclusters)
    for cluster in range(nclusters):
            print("cluster ",cluster,":")
            for i,sentence in enumerate(clusters[cluster]):
                    print("\tsentence ",i,": ",sentences[sentence])