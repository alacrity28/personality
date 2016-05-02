from model import model
from sklearn.decomposition import NMF, FactorAnalysis, PCA, ProjectedGradientNMF, TruncatedSVD, FastICA
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


def nmf(answers_clean, factors = 5, top_questions = 10, iterations = 2):

    final = []
    master_dict = {}

    nmf = NMF(n_components = factors)
    W = nmf.fit_transform(answers_clean)
    H = nmf.components_
    for j in xrange(factors):
        master_dict[j] = Counter([i for i in H[j].argsort()[:-top_questions - 1:-1]])

    init_dict = [[i for i in values] for key, values in master_dict.iteritems()]
    #print "initialize master_dict:", master_dict


    for iter_ in xrange(iterations-1):

        nmf = NMF(n_components = factors)
        W = nmf.fit_transform(answers_clean)
        H = nmf.components_

        for j in xrange(factors):
            '''
            takes a row from the new W matrix to figure out where it should go
            '''
            new_set = set([i for i in H[j].argsort()[:-top_questions - 1:-1]])
            #print "new_set", new_set

            best_intersection = set()
            best_match_key = 1000

            for key, value in master_dict.iteritems():
                '''
                searching for the best match
                '''
                most_common_values = set([k[0] for k in value.most_common(10)])
                #print "most_common_values {} for key {}".format(most_common_values, key)

                temp_intersection = most_common_values.intersection(new_set)
                #print "temp_intersection", temp_intersection

                if len(temp_intersection) > len(best_intersection):
                    best_intersection = temp_intersection
                    best_match_key = key

                #print "best_intersection", best_intersection
                #print "..................................."

            #print "best_match_key", best_match_key
            #print
            master_dict[best_match_key].update(new_set)

            set_count = 0


        for key,value in master_dict.iteritems():
            initial = set(init_dict[key])
            most_common = set([k[0] for k in value.most_common(10)])
            set_count += len(initial.intersection(most_common))
            if iterations == iter_ + 2:
                print "inital:", sorted(initial)
                print "final:", sorted(most_common)
                print
        final.append(set_count)


        print "iteration:", iter_

    return init_dict, master_dict



    '''
    1.) NMF
    2.) take the summarized factors and turn in to quiz
    3.) take the top x# factors from simple nmf and turn into short quiz, how do we go back up in dimensionality,
    smoting?
    4.) how do I score the quiz? probability distribution (assume a normal distribution) or a kernel desnity estimation (histogram smoother)
    sort the scores, take the persons score in the factor space and figure out where they lie?, count below/above

    '''



if __name__ == '__main__':

    main_ = model()
    data, questions, country_dict, questions_dict, answers_clean, answers_messy = main_.load_data()

    final = nmf(answers_clean, iterations = 1000)
