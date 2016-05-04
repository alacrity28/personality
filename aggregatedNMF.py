from model import model
from sklearn.decomposition import NMF, FactorAnalysis, PCA, ProjectedGradientNMF, TruncatedSVD, FastICA
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


def aggregated_nmf(answers_clean, factors = 5, top_questions = 10, iterations = 2):
    '''
    Input: dataframe, integer, integer, integer
    Output: dictionary, dictionary

    This function runs multiple iterations of nmf and for each iteration,
    it matches the top 10 questions of that factor with the top 10 overall most
    referenced questions for each factor in the master dictionary. It then adds those 10 ten questions with the closest match.
    This ensemble NMF cleans the NMF factors because individual runs of nmf often produce similar factors but using slightly different
    questions. The aggregated NMF finds the most common questions used to describe each factor.
    '''

    final = []
    master_dict = {}

    #runs one intial nmf and enters that data in to a master_dict which will hold the top 10 questions for each factor for each iteration
    nmf = NMF(n_components = factors)
    W = nmf.fit_transform(answers_clean)
    H = nmf.components_
    for j in xrange(factors):
        master_dict[j] = Counter([i for i in H[j].argsort()[:-top_questions - 1:-1]])

    #creates a dictionary to save the first iteration of nmf so that it can be referenced later for analysis (not neccesary to the actual model)
    init_dict = [[i for i in values] for key, values in master_dict.iteritems()]

    #For n-1 iterations (because we already ran the first one), run the nmf and assign the resulting factors to their closest master factor
    for iter_ in xrange(iterations-1):

        #run a new nmf
        nmf = NMF(n_components = factors)
        W = nmf.fit_transform(answers_clean)
        H = nmf.components_

        #this long loop takes each new factor produced from the nmf and figures out which master factor (in master_dict) it should fall in to
        for j in xrange(factors):

            #takes the top 10 questions for the jth new factor produced from the nmf in the most recent iteration
            new_set = set([i for i in H[j].argsort()[:-top_questions - 1:-1]])

            #defines an empty set and huge match key which will be adjusted as we iterate through to search for the best match
            best_intersection = set()
            best_match_key = 1000

            for key, value in master_dict.iteritems():
                #iterates through the master dictionary to figure out which master factor's most frequent 10 questions best match the new_set's top 10

                most_common_values = set([k[0] for k in value.most_common(10)])
                temp_intersection = most_common_values.intersection(new_set)

                #if the number of new overlapping factors is larger than the
                #current best, replace the set and reset the key which denotes which factor has the current best match
                if len(temp_intersection) > len(best_intersection):
                    best_intersection = temp_intersection
                    best_match_key = key

            #once we have found the best match, update the master dict with the top 10 questions from that factor
            master_dict[best_match_key].update(new_set)

        #this section is for reporting purposes, it records the total number of itersections between the master_dict and the factors in a new nmf
        #for each round, it enables us to look at convergence of the model
            set_count = 0
        for key,value in master_dict.iteritems():
            initial = set(init_dict[key])
            most_common = set([k[0] for k in value.most_common(10)])
            set_count += len(initial.intersection(most_common))
            if iterations == iter_ + 2:
                print "inital:", sorted(initial)
                print "final:", sorted(most_common)
                print

        #once the set count (total number of itersections per nmf round) is calculated, it is appended to final to analysis later
        final.append(set_count)

        print "iteration:", iter_

    return init_dict, master_dict


if __name__ == '__main__':

    main_ = model()
    data, questions, country_dict, questions_dict, answers_clean, answers_messy = main_.load_data()

    #output = aggregated_nmf(answers_clean, iterations = 1000)
