import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.decomposition import ProjectedGradientNMF
from numpy import array, matrix, linalg
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from incf.countryutils import transformations
import json
import csv


def import_and_clean():

    #import data
    data = pd.read_csv('data/data.csv', delimiter = "\t")
    meta_data = pd.read_csv('data/questions', sep='\t')
    country = pd.read_csv('data/country', sep='\t', delimiter = ",", dtype = {"Code": np.str, "Country": np.str}) #index_col = 'Code'

    #clean_meta_data
    questions, questions_dict = extract_questions(meta_data, data)

    #country and continent
    #make country dict
    country_dict = create_country_dict(country)


    #add long country anme to data
    data['country_name'] = data['country'].apply(lambda x: country_dict[x])    #trait_pre1.applymap(lambda x: change_trait(x))
    data['Continent'] = data['country'].apply(lambda x: get_continent(x))


    #create age bucket
    data = create_age_bucket(data)

    #create just the answers dataframe
    clean_answers, clean_data, messy_answers = create_answers(data, questions)

    #answers_correct_axis = clean_answers.apply(lambda x: adjust_scale(x, questions))


    with open('data/clean_data/country_dict', 'w') as outfile:
        json.dump(country_dict, outfile)

    with open('data/clean_data/questions_dict', 'w') as outfile:
        json.dump(questions_dict, outfile)

    clean_data.to_csv('data/clean_data/data.csv')
    questions.to_csv('data/clean_data/questions.csv')
    clean_answers.to_csv('data/clean_data/answers_clean.csv')
    messy_answers.to_csv('data/clean_data/messy_answers.csv')

    #return answers_correct_axis, answers_correct_axis.shape

    return "Data has been cleaned and exported!!!! Good for you!!"


def extract_questions(meta_data, data):

    #This code insert's the question that is missing from the codebook but is in the dataset: P10 - "I have a good word for everyone." This appropriate index is preserved (index = 162).
    first = meta_data[0:162]
    last = meta_data[162:]
    new_row = pd.DataFrame([['P10','INTEGER', 'I have a good word for everyone.']], columns=['Field', 'Format', 'Description'])
    first = first.append(new_row, ignore_index=True)
    meta_data = first.append(last, ignore_index=True)

    questions_raw = meta_data[0:163][['Field','Description']]
    questions_raw['Description'] = questions_raw['Description'].apply(lambda x: x.split(" rated")[0])
    questions_extracted = pd.DataFrame(questions_raw)

    #adding variables
    questions_extracted['Trait_Letter'] = [x[0] for x in data.columns[:163]]
    questions_extracted['Trait'] = questions_extracted['Trait_Letter'].apply(lambda x: change_trait(x))

    ct_a = [1,] * 7 + [0,] * 3
    ct_b = [1,] * 8 + [0,] * 5
    ct_c = [1,] * 5 + [0,] * 5
    ct_d = [1,] * 6 + [0,] * 4
    ct_e = [1,] * 6 + [0,] * 4
    ct_f = [1,] * 5 + [0,] * 5
    ct_g = [1,] * 5 + [0,] * 5
    ct_h = [1,] * 6 + [0,] * 4
    ct_i = [1,] * 6 + [0,] * 4
    ct_j = [1,] * 7 + [0,] * 3
    ct_k = [1,] * 5 + [0,] * 5
    ct_l = [1,] * 8 + [0,] * 2
    ct_m = [1,] * 5 + [0,] * 5
    ct_n = [1,] * 7 + [0,] * 3
    ct_o = [1,] * 5 + [0,] * 5
    ct_p = [1,] * 7 + [0,] * 3
    final = ct_a + ct_b + ct_c + ct_d + ct_e + ct_f + ct_g + ct_h + ct_i + ct_j + ct_k + ct_l + ct_m + ct_n + ct_o + ct_p
    questions_extracted['Coding'] = questions_extracted.index.map(lambda x: final[x])


    #making questions_dict
    questions_dict = {}
    for i in questions_extracted.iterrows():
        questions_dict[i[1][2].lower()] = i[1][3]

    return questions_extracted, questions_dict


def create_country_dict(country):
    country_dict = {}
    for i in country.iterrows():
        country_dict[i[1][0]] = i[1][1]
    country_dict['FX'] = 'France, Metropolitan'
    return country_dict


def create_age_bucket(data, bins = [0, 20, 30, 40, 50, 60, 70, 100], group_names = ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']):

    data['age_bucket'] = pd.cut(data['age'], bins, labels=group_names)
    return data


def create_answers(data, questions):

    just_answers = [i for i in questions['Field']]
    clean_data = data[(data[just_answers] == 0).sum(axis=1) < 2]

    #this gets just the answers that have 1 or no missing values
    messy_answers = clean_data[just_answers]

    #column means of the cleaned data
    column_means = messy_answers.mean(axis=0)

    zero_to_nan = messy_answers.replace(0, np.nan)
    clean_answers = zero_to_nan.fillna(column_means)

    return clean_answers, clean_data, messy_answers


def adjust_scale(row, questions):

    changed_list = []
    for key, value in row.iteritems():
        if all(questions[questions['Field'] == key]['Coding'] == 0):
            changed_list.append(change_scale(value[0]))
        else:
            changed_list.append(value[0])
    row = pd.Series(changed_list)
    return row


def remove_nan(answers):

    clean_answers = answers[(answers == 0).sum(axis=1) < 2]
    column_means = clean_answers.mean(axis=0)
    no_zeros = clean_answers.replace(0, np.nan)
    no_zeros = no_zeros.fillna(column_means)

    return no_zeros

def change_scale(value):
    new = 100

    if value == 5:
        new = 1
    elif value == 4:
        new = 2
    elif value == 2:
        new = 4
    elif value == 1:
        new = 5
    elif value == 3:
        new = 3
    elif value == 0:
        new = 0
    else:
        new = 100
    return new


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


def total_questions_in_category(x):

    if x == 'A':
        return 10
    if x == 'B':
        return 13
    if x == 'C':
        return 10
    if x == 'D':
        return 10
    if x == 'E':
        return 10
    if x == 'F':
        return 10
    if x == 'G':
        return 10
    if x == 'H':
        return 10
    if x == 'I':
        return 10
    if x == 'J':
        return 10
    if x == 'K':
        return 10
    if x == 'L':
        return 10
    if x == 'M':
        return 10
    if x == 'N':
        return 10
    if x == 'O':
        return 10
    if x == 'P':
        return 9


def get_continent(country_short):

    #country abreiviation
    try:
        continent = transformations.cca_to_ctn(country_short)
    except:
        if country_short == "EU":
            continent = "Europe"
        elif country_short == "AP":
            continent = "Asia"
        else:
            continent = "none"
    return continent


if __name__ == '__main__':
    print import_and_clean()
