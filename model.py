import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, FactorAnalysis, PCA, ProjectedGradientNMF, TruncatedSVD, FastICA
from numpy import array, matrix, linalg
from sklearn.metrics import mean_squared_error
from scipy.linalg import svd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind
import json
from scipy.linalg import svd
import cPickle


class model(object):

    def __init__(self):
        '''
        Input: instance variables
        Output: Nothing

        This function initializes cleaned data (from the load_data function) that was read in, cleaned and saved ouy by cleaning.py.
        '''

        data, questions, country_dict, questions_dict, answers_clean, messy_answers = self.load_data()

        self.data = data
        self.questions = questions
        self.messy_answers = messy_answers
        self.country_dict = country_dict
        self.questions_dict = questions_dict
        self.answers_clean = answers_clean

    def load_data(self):
        '''
        Input: instance variable
        Output: dataframe, dataframe, dictionary, dictionary, dataframe, dataframe

        This function loads in the data that was cleaned in cleaning.py.
        '''

        with open('data/clean_data/questions_dict') as json_file:
            questions_dict = json.load(json_file)

        with open('data/clean_data/country_dict') as json_file:
            country_dict = json.load(json_file)

        data = pd.read_csv("data/clean_data/data.csv").drop('Unnamed: 0', axis = 1)
        questions = pd.read_csv("data/clean_data/questions.csv").drop('Unnamed: 0', axis = 1)
        answers_clean = pd.read_csv("data/clean_data/answers_clean.csv").drop('Unnamed: 0', axis = 1)
        messy_answers = pd.read_csv("data/clean_data/messy_answers.csv").drop('Unnamed: 0', axis = 1)

        return data, questions, country_dict, questions_dict, answers_clean, messy_answers

    def pull_factors(self, col_name):
        '''
        Input: Instance Variable, string
        Output: dataframe

        This function was used to easily pull out the groupings of questions that the survey questions are formatted in. This
        is a helper function used for easy EDA analysis.
        '''

        len_ = len(col_name)
        check_value = questions_dict[col_name.lower()] if len_ > 1 else col_name
        good_cols = [col for col in self.answers_clean.columns if col[0] == check_value]
        return self.answers_clean[good_cols]

    def reconst_mse(self, target, left, right):
        '''
        Input: instance variable, dataframe, dataframe, dataframe
        Output: float

        This function calculates the Reconstruction mean squared error.
        '''
        return (array(target - left.dot(right))**2).mean()

    def describe_results(self, W, H, top_questions = 10):
        '''
        Input: instance variable, dataframe, dataframe, integer
        Output: nothing

        This prints out the top 10 questions related to the topics found in the model.
        It is used to interpret the model results.
        '''
        print("Reconstruction error: %f") %(self.reconst_mse(self.answers_clean, W, H))
        for topic_num, topic in enumerate(H):
            print("Topic %d:" % topic_num)
            print [self.questions.loc[i]['Description'] for i in topic.argsort()[:-top_questions - 1:-1]]
            print [self.questions.loc[i]['Field'] for i in topic.argsort()[:-top_questions - 1:-1]]


    def nmf(self, factors = 5, top_questions = 10):
        '''
        Input: instance variable, integer, integer
        Output: dataframe, dataframe, object

        This function performs NMF, calls the describe function which returns the results and returns the W and H matrix.
        It also returns the nmf model itself so that other functions can be called on it during EDA.
        '''
        nmf = NMF(n_components = factors)
        W_nmf = nmf.fit_transform(self.answers_clean)
        H_nmf = nmf.components_
        self.describe_results(W_nmf, H_nmf, top_questions)
        return W_nmf, H_nmf, nmf

    def pca(self, factors = 5, top_questions = 10):
        '''
        Input: instance variable, integer, integer
        Output: dataframe, dataframe, list, integer

        This function performs PCA, calls the describe function which returns the results and returns the decomposed matricies along with
        other m
        '''
        pca = PCA(n_components = factors)
        W = pca.fit_transform(self.answers_clean)
        H = pca.components_
        var_ratio = pca.explained_variance_ratio_
        covariance_ = pca.get_covariance()
        self.describe_results(W, H, top_questions)
        return W, H, var_ratio, covariance_

    def ica(self, factors = 5, top_questions = 10):
        '''
        Input: instance variable, integer, integer
        Output: dataframe, dataframe, list, integer

        This function performs ICA, calls the describe function which returns the results and returns the decomposed matricies.
        '''

        ica = FastICA(n_components = factors)
        W = ica.fit_transform(self.answers_clean)
        H = ica.components_
        mixing_ = ica.mixing_
        n_iter_ = ica.n_iter_
        self.describe_results(W, H, top_questions)
        return W, H, mixing_, n_iter_


    def svd(self, factors = 5, top_questions = 10):
        '''
        Input: instance variable, integer, integer
        Output: dataframe, dataframe, list, integer

        This function performs SVD, calls the describe function which returns the results and returns the decomposed matricies.
        It also returns the svd model itself so that other functions can be called on it during EDA.
        '''

        svd = TruncatedSVD(n_components = factors)
        W = svd.fit_transform(self.answers_clean)
        H = svd.components_
        var_ratio = svd.explained_variance_ratio_
        var = svd.explained_variance_
        self.describe_svd_results(W, H, top_questions)
        return W, H, var_ratio, var

    def describe_svd_results(self, W, H, top_questions = 10):
        '''
        Input: instance variable, dataframe, dataframe, integer
        Output: nothing

        This prints out the top 10 questions related to the topics found in the svd model.
        It is used to interpret the model results.
        '''
        for topic_num, topic in enumerate(H):
            print("Topic %d:" % topic_num)
            print [self.questions.loc[i]['Description'] for i in topic.argsort()[:-top_questions - 1:-1]]
            print [self.questions.loc[i]['Field'] for i in topic.argsort()[:-top_questions - 1:-1]]

    def kmeans(self, clusters = 5):
        '''
        Input: instance variable, integer
        Output: nothing

        Fits a kmeans model and calls the describe function.
        '''
        kmeans = KMeans(n_clusters = clusters)
        kmeans.fit(self.answers_clean)
        self.describe_kmeans_results(kmeans)

    def describe_kmeans_results(self, kmeans):
        '''
        Input: instance variable, integer
        Output: nothing

        Prints the top 10 question closest to the center of a cluster.
        '''

        top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
        print "top features for each cluster:"
        for num, centroid in enumerate(top_centroids):
            print "%d:" % (num)
            print [self.questions.loc[i]['Description'] for i in centroid]
            print [self.questions.loc[i]['Field'] for i in centroid]
            print "................................................................"


    def pickle(self):
        '''
        Input: instance variable
        Output: nothing

        Performs 3 nmf models with 3 different component levels. Then it pickles the data and model.
        Because NMF produces non-unique results, this function runs an nmf and freezes the results so that I
        can reference the same solution to the nmf at different occasions.
        '''

        W_5, H_5, nmf_5 = self.nmf(factors = 5, top_questions = 10)
        W_8, H_8, nmf_8 = self.nmf(factors = 8, top_questions = 10)
        W_16, H_16, nmf_16 = self.nmf(factors = 16, top_questions = 10)

        cPickle.dump( W_5, open( "data/clean_data/W_5.p", "wb" ) )
        cPickle.dump( H_5, open( "data/clean_data/H_5.p", "wb" ) )
        cPickle.dump( nmf_5, open( "data/clean_data/nmf_5.p", "wb" ) )

        cPickle.dump( W_8, open( "data/clean_data/W_8.p", "wb" ) )
        cPickle.dump( H_8, open( "data/clean_data/H_8.p", "wb" ) )
        cPickle.dump( nmf_8, open( "data/clean_data/nmf_8.p", "wb" ) )

        cPickle.dump( W_16, open( "data/clean_data/W_16.p", "wb" ) )
        cPickle.dump( H_16, open( "data/clean_data/H_16.p", "wb" ) )
        cPickle.dump( nmf_16, open( "data/clean_data/nmf_16.p", "wb" ) )

        print "The pickling is complete! Yummy!"


if __name__ == '__main__':
    data, questions, answers, country_dict, questions_dict = load_data()

    print "data loaded!!!!"
