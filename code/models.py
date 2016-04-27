import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import NMF
from sklearn.decomposition import ProjectedGradientNMF
from numpy import array, matrix, linalg
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


class Personality(filename = 'data/16PF/data.csv', data_delimiter = "\t", metadata = 'data/16PF/questions', meta_sep='\t', country_data ='data/16PF/country')


    def import():
        self.data = pd.read_csv(filename, delimiter = data_delimiter)
        self.meta_data = clean_meta_data(metadata, meta_sep))
        self.country = pd.read_csv(country_data, sep='\t', delimiter = ",", dtype = {"Code": np.str, "Country": np.str}) #index_col = 'Code'
        self.country_dict = create_country_dict()
        self.questions_dict = {}
        self.answers = create_answers()


    def import_and_clean():

        #import data
        data = pd.read_csv('data/16PF/data.csv', delimiter = "\t")
        meta_data = pd.read_csv('data/16PF/questions', sep='\t')
        country = pd.read_csv('data/16PF/country', sep='\t', delimiter = ",", dtype = {"Code": np.str, "Country": np.str}) #index_col = 'Code'

        #clean_meta_data
        questions = clean_questions(meta_data)

        #country
        #make country dict
        country_dict = create_country_dict()
        #add long country anme to data
        data['country_name'] = data['country'].apply(lambda x: self.country_dict[x])    #trait_pre1.applymap(lambda x: change_trait(x))

        #clean_questions
        questions, questions_dict = format_questions(meta_data)

        #create age bucket
        data = create_age_bucket(data)




    def clean_meta_data(meta_data):
        questions_raw = meta_data[0:162][['Field','Description']]
        questions_raw['Description'] = questions_raw['Description'].apply(lambda x: x.split(" rated")[0])
        #questions = questions_raw.set_index(questions_raw['Field'])['Description'].apply(lambda x: x.split(" rated")[0])
        questions = pd.DataFrame(questions)
        # Do I need to initialize this earlier?
        return questions

    def create_country_dict():
        country_dict = {}
        for i in self.country.iterrows():
            country_dict[i[1][0]] = i[1][1]
        country_dict['FX'] = 'France, Metropolitan'
        return country_dict


    def format_questions(meta_data):

        category_columns = [x[0] for x in meta_data.columns[:163]
        trait_letter = pd.DataFrame(category_columns, columns = ['trait_letter'], index = meta_data.columns[0:163])
        questions['Trait_Letter'] = trait_letter
        questions['Trait'] = trait_letter.applymap(lambda x: change_trait(x))
        questions = questions[:-1]

        questions_dict = {}
        for i in questions.iterrows():
            questions_dict[i[1][1].lower()] = i[1][2]

        return questions, questions_dict


    def create_age_bucket(data, bins = [0, 20, 30, 40, 50, 60, 70, 120], group_names = ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']):
        self.data['age_bucket'] = pd.cut(data['age'], bins, labels=group_names)
        return data


    def create_answers():
        exclude = ['age','gender','accuracy','country','source','elapsed','country_name','age_bucket']
        return self.data.ix[:, self.data.columns.difference(exclude)]


    def change_trait(x):
        if x == 'A':
            return "Warmth"
        if x == 'B':
            return "Reasoning"
        if x == 'C':
            return "Emotional Stability"
        if x == 'D':
            return "Dominance"
        if x == 'E':
            return "Liveliness"
        if x == 'F':
            return "Rule-Consciousness"
        if x == 'G':
            return "Social Boldness"
        if x == 'H':
            return "Sensitivity"
        if x == 'I':
            return "Vigilance"
        if x == 'J':
            return "Abstractedness"
        if x == 'K':
            return "Privateness"
        if x == 'L':
            return "Apprehension"
        if x == 'M':
            return "Openness to Change"
        if x == 'N':
            return "Self-Reliance"
        if x == 'O':
            return "Perfectionism"
        if x == 'P':
            return "Tension"


    def pull_factors(df, col_name):
        len_ = len(col_name)
        check_value = questions_dict[col_name.lower()] if len_ > 1 else col_name
        good_cols = [col for col in df.columns if col[0] == check_value]
        return df[good_cols]


    def reconst_mse(target, left, right):
        return (array(target - left.dot(right))**2).mean()

    def describe_nmf_results(document_term_mat, W, H, n_top_words = 15):
        print("Reconstruction error: %f") %(reconst_mse(document_term_mat, W, H))
        for topic_num, topic in enumerate(H):
            print("Topic %d:" % topic_num)
            #print(" ".join([questions[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print [questions_raw.loc[i][1] for i in topic.argsort()[:-n_top_words - 1:-1]]
            print [questions_raw.loc[i][0] for i in topic.argsort()[:-n_top_words - 1:-1]]

            if topic_num > 10:
                break
