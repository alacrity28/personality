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

        data, questions, country_dict, questions_dict, answers_clean, messy_answers = self.load_data()

        self.data = data
        self.questions = questions
        self.messy_answers = messy_answers
        self.country_dict = country_dict
        self.questions_dict = questions_dict
        self.answers_clean = answers_clean

    def load_data(self):

        with open('data/clean_data/questions_dict') as json_file:
            questions_dict = json.load(json_file)

        with open('data/clean_data/country_dict') as json_file:
            country_dict = json.load(json_file)

        data = pd.read_csv("data/clean_data/data.csv").drop('Unnamed: 0', axis = 1)
        questions = pd.read_csv("data/clean_data/questions.csv").drop('Unnamed: 0', axis = 1)
        answers_clean = pd.read_csv("data/clean_data/answers_clean.csv").drop('Unnamed: 0', axis = 1)
        messy_answers = pd.read_csv("data/clean_data/messy_answers.csv").drop('Unnamed: 0', axis = 1)

        return data, questions, country_dict, questions_dict, answers_clean, messy_answers


    #helper functions
    def pull_factors(self, col_name):

        len_ = len(col_name)
        check_value = questions_dict[col_name.lower()] if len_ > 1 else col_name
        good_cols = [col for col in self.answers_clean.columns if col[0] == check_value]
        return self.answers_clean[good_cols]

    '''
    def group_letter(answers):
    return answers[0]

    answers.groupby(group_letter, axis = 1).mean()
    '''
    def reconst_mse(self, target, left, right):
        return (array(target - left.dot(right))**2).mean()

    def describe_results(self, W, H, top_questions = 10):
        print("Reconstruction error: %f") %(self.reconst_mse(self.answers_clean, W, H))
        for topic_num, topic in enumerate(H):
            print("Topic %d:" % topic_num)
            print [self.questions.loc[i]['Description'] for i in topic.argsort()[:-top_questions - 1:-1]]
            print [self.questions.loc[i]['Field'] for i in topic.argsort()[:-top_questions - 1:-1]]

    def nmf(self, factors = 5, top_questions = 10):
        nmf = NMF(n_components = factors)
        W_nmf = nmf.fit_transform(self.answers_clean)
        H_nmf = nmf.components_
        self.describe_results(W_nmf, H_nmf, top_questions)
        #return self.reconst_mse(self.answers_clean, W_nmf, H_nmf)
        return W_nmf, H_nmf, nmf

    def svd(self, factors = 5, top_questions = 10):
        svd = TruncatedSVD(n_components = factors)
        W = svd.fit_transform(self.answers_clean)
        H = svd.components_
        var_ratio = svd.explained_variance_ratio_
        var = svd.explained_variance_
        self.describe_results(W, H, top_questions)
        #reconstructed = self.reconst_mse(self.answers_clean, W, H)
        #return reconstructed
        return W, H, var_ratio, var

    def pca(self, factors = 5, top_questions = 10):
        pca = PCA(n_components = factors)
        W = pca.fit_transform(self.answers_clean)
        H = pca.components_
        var_ratio = pca.explained_variance_ratio_
        covariance_ = pca.get_covariance()
        self.describe_results(W, H, top_questions)
        #reconstructed = self.reconst_mse(self.answers_clean, W, H)
        #return reconstructed
        return W, H, var_ratio, covariance_

    def ica(self, factors = 5, top_questions = 10):
        ica = FastICA(n_components = factors)
        W = ica.fit_transform(self.answers_clean)
        H = ica.components_
        mixing_ = ica.mixing_
        n_iter_ = ica.n_iter_
        #self.describe_results(W, H, top_questions)
        reconstructed = self.reconst_mse(self.answers_clean, W, H)
        return reconstructed
        #return W, H, mixing_, n_iter_

    #kmeans
    def kmeans(self, clusters = 5):
        kmeans = KMeans(n_clusters = clusters)
        kmeans.fit(self.answers_clean)
        self.describe_kmeans_results(kmeans)

    def describe_kmeans_results(self, kmeans):
        top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
        print "top features for each cluster:"
        for num, centroid in enumerate(top_centroids):
            print "%d:" % (num)
            print [self.questions.loc[i]['Description'] for i in centroid]
            print [self.questions.loc[i]['Field'] for i in centroid]
            print "................................................................"


    def pickle(self):

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

    '''
    /code to load from pickle

    W_5 = cPickle.load( open( "data/clean_data/W_5.p", "rb" ) )
    H_5 = cPickle.load( open( "data/clean_data/H_5.p", "rb" ) )
    nmf_5 = cPickle.load( open( "data/clean_data/nmf_5.p", "rb" ) )

    W_8 = cPickle.load( open( "data/clean_data/W_8.p", "rb" ) )
    H_8 = cPickle.load( open( "data/clean_data/H_8.p", "rb" ) )
    nmf_8 = cPickle.load( open( "data/clean_data/nmf_8.p", "rb" ) )

    W_16 = cPickle.load( open( "data/clean_data/W_16.p", "rb" ) )
    H_16 = cPickle.load( open( "data/clean_data/H_16.p", "rb" ) )
    nmf_16 = cPickle.load( open( "data/clean_data/nmf_16.p", "rb" ) )
    '''

'''
def factor_analysis(self, n_components = 16, svd_method = 'lapack'):
    #lapack is the extact same as SVD from scipy.linalg
    fa = FactorAnalysis(n_components = n_components, svd_method = svd_method)
    fa = fit_transform(self.data)
    describe_factor_analysis(fa)

def describe_factor_analysis(self, fa):
    pass
'''

'''
    fit(X[, y])	Fit the FactorAnalysis model to X using EM
fit_transform(X[, y])	Fit to data, then transform it.
get_covariance()	Compute data covariance with the FactorAnalysis model.
get_params([deep])	Get parameters for this estimator.
get_precision()	Compute data precision matrix with the FactorAnalysis model.
score(X[, y])	Compute the average log-likelihood of the samples
score_samples(X)	Compute the log-likelihood of each sample
set_params(**params)	Set the parameters of this estimator.
transform(X)	Apply dimensionality reduction to X using the model.
'''

###other models
#scipy.stats.ttest_ind
'''
#gets the mean gender
mean_gender = data.groupby('gender').mean()[answers.columns]

#takes a column name and returns the letter related to it
def group_letter(answers):
    return answers[0]

mean_gender.groupby(group_letter, axis = 1).mean()

men = data[data['gender'] == 1][answers.columns]
women = data[data['gender'] == 2][answers.columns]

men_factors = men.groupby(group_letter, axis = 1).mean()
women_factors = women.groupby(group_letter, axis = 1).mean()

for i in questions_dict.keys():
    print "{0}: {1}".format(str(i.upper()), ttest_ind(men_factors[str(i.upper())], women_factors[str(i.upper())])[1])


for i in questions.iterrows():
    print "{0}: {1} - {2}".format(str(i[1][0].upper()), i[1][1], ttest_ind(men[men[str(i[1][0].upper())] != 0][str(i[1][0].upper())], women[women[str(i[1][0].upper())] != 0][str(i[1][0].upper())])[1] < .01)
    print ""

plt.bar(range(len(gender_means[1:2].columns.values)), gender_means[1:2].values.T, tick_label=gender_means[1:2].columns, align = 'center', color = '#')
plt.bar(range(len(gender_means[2:3].columns.values)), gender_means[2:3].values.T, tick_label=gender_means[2:3].columns, align = 'center', color = '#')


elbow/plot:
errors = []
for i in xrange(1,21):
    error = main_.nmf(factors = i, top_questions = 8)
    errors.append(error)
plt.plot(errors)

'''

if __name__ == '__main__':
    data, questions, answers, country_dict, questions_dict = load_data()

    print "data loaded!!!!"
