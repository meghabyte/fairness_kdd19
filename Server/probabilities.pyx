import fairnessMetrics
import itertools
from heapq import *
import numpy as np
import math
import copy

fairness_theories = fairnessMetrics.fairness_theories
usegini = False
usedeep = False
max_h_index = 4
items_truelabel_list = [
        [
            np.array([
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0]
            ]), 
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
        ],
        [
            np.array([
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0]
            ]),
            [0, 1, 0, 0, 0, 0, 0, 1, 1, 0]
        ],
        [
            np.array([
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0]
            ]),
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        ],
        [
            np.array([
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0]
            ]), 
            [0, 0, 0, 1, 1, 0, 1, 1, 0, 1]
        ]
    ]
#items_truelabel_list = fairnessTests.random_items_generator(10, 2)


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def observe_test(test, case_chosen, test_observations):
    new_test_observations = copy.deepcopy(test_observations)
    new_test_observations.append([test, case_chosen, dict()])
    #new_test_observations.append((test, case_chosen))
    return new_test_observations


#P(Test=x | Observations)
def p_outcome(demographics_indices, dem_values, observations, test, case_chosen, p_h_obs=None):
    if isinstance(case_chosen,list):
        p = [0.0]*len(case_chosen)
        for h in range(0,max_h_index):
            x = priors(demographics_indices, dem_values, test, h)
            for c in case_chosen:
                if p_h_obs==None:
                    p[c] += x[c] * p_h_observations(demographics_indices, dem_values, observations, h)
                else:
                    p[c] += x[c] * p_h_obs[h]
    else:
        p = 0.0
        #HYPTOHESIS INDICES
        for h in range(0,max_h_index):
            if p_h_obs==None:
                p += (priors(demographics_indices, dem_values, test, h)[case_chosen] * p_h_observations(demographics_indices, dem_values, observations, h))
            else:
                p += (priors(demographics_indices, dem_values, test, h)[case_chosen] * p_h_obs[h])
    return p


#P(Observations)
def p_observations(demographic_indices, dem_values, test_observations):
    cdef double sum = 0.0
    #HYPOTHESIS_INDICES
    for h in range(0,max_h_index):
        sum = sum + (p_observations_h(demographic_indices, dem_values, test_observations, h) * p_h())
    return sum


#P(Fairness Metric)
cdef double p_h():
    #HYPOTHESIS_INDICES
    return 1.0/max_h_index#1.0/len(fairness_theories)


#P(Fairness Metric | Observations+Test)
cdef double p_h_observations_with_test(demographic_indices, dem_values, test_observations, hypothesis_indx, test, case_chosen, p_obs=None):
    if usedeep:
        observations_with_test = observe_test(test, case_chosen, test_observations) 
        p_obs_with_test = p_observations(demographic_indices, dem_values, observations_with_test)
        old_answer =  (p_observations_h(demographic_indices, dem_values, observations_with_test, hypothesis_indx) * p_h())/p_obs_with_test
        answer = old_answer
    else:
        test_observations.append([test, case_chosen, dict()])
        p_obs_with_test = p_observations(demographic_indices, dem_values, test_observations)
        new_answer = (p_observations_h(demographic_indices, dem_values, test_observations, hypothesis_indx) * p_h())/p_obs_with_test 
        answer = new_answer
        test_observations.pop()
    return answer


#P(Fairness Metric | Observations)
def p_h_observations(demographic_indices, dem_values, test_observations, hypothesis_indx, p_obs=None):
    if (len(test_observations) == 0):
        print("Empty Observations")
        return -1.0
    if p_obs==None:
        p_obs = p_observations(demographic_indices, dem_values, test_observations)
    return (p_observations_h(demographic_indices, dem_values, test_observations, hypothesis_indx) * p_h())/p_obs


#P(Observations | Fairness Metric)
def p_observations_h(demographic_indices, dem_values, test_observations, hypothesis_indx):
    if len(test_observations)==0:
        return 1.0
    else:
        obs = test_observations[-1]
        if hypothesis_indx in obs[2]:
            return obs[2][hypothesis_indx]
        priors_tuple = priors(demographic_indices, dem_values, obs[0], hypothesis_indx)
        priors_tuple_index = obs[1]
        if len(test_observations)>1:
            curr_prob = test_observations[-2][2][hypothesis_indx]
        else:
            curr_prob = 1.0
        curr_prob = curr_prob * priors_tuple[priors_tuple_index]
        obs[2][hypothesis_indx] = curr_prob
        return curr_prob


# remove some redundancy - this is fastest
def generalized_entropy(inequalities):
    mean = np.mean([i for i in inequalities if (i >= 0 and i <= 1)])
    new_inequalities = [i if (i >= 0 and i <= 1) else mean for i in inequalities]
    fraction = 1.0/(len(new_inequalities) * 2.0)
    summation = np.sum([math.pow((yi*1.0/mean),2)-1.0 for yi in new_inequalities])
    #assert np.abs(fraction * summation - generalized_entropy_1(inequalities))<1.0E-5
    return fraction * summation


cdef double gini(inequalities):
    new_inequalities = [i for i in inequalities if (i >= 0 and i <= 1)]
    #print("There are new inequalities of length: "+str(len(new_inequalities)))
    return sum([abs(x-y) for x in new_inequalities for y in new_inequalities]) / (2.0*len(new_inequalities)*sum(new_inequalities))

#P(Observation | Fairness Metric)
def priors(demographics_indices, dem_values, test, hypothesis_indx):
    if len(test)==4:
        return (test[3][hypothesis_indx][0],test[3][hypothesis_indx][1])
    case1_pl = test[2][0]["pl"]
    case2_pl = test[2][1]["pl"]
    case1_tl = test[2][0]["tl"]
    case2_tl = test[2][1]["tl"]
    items = items_truelabel_list[test[2][0]["item_idx"]][0].tolist()
    #values_permutations = list(itertools.product(*dem_values))
    #print("Values: " + str(values_permutations))
    case1_inequalities = []
    case2_inequalities = []
    #for val_tuple in values_permutations:
    for val_tuple in itertools.product(*dem_values):
        #print("PRIORS: CURRENT VAL TUPLE: " + str(val_tuple))
        #filtered_item_ind = []
        filtered_case1_pl = []
        filtered_case2_pl = []
        filtered_case1_tl = []
        filtered_case2_tl = []
        filtered_items = []
        for item_indx, item in enumerate(items):
            put_item = True
            for list_index, dem_indx in enumerate(demographics_indices):
                if (item[dem_indx] != val_tuple[list_index]):
                    put_item = False
                    break
            if (put_item == True):
                #filtered_item_ind.append(item_indx)
                filtered_case1_pl.append(case1_pl[item_indx])
                filtered_case2_pl.append(case2_pl[item_indx])
                filtered_case1_tl.append(case1_tl[item_indx])
                filtered_case2_tl.append(case2_tl[item_indx])
                filtered_items.append(items[item_indx])            
        case1_inequalities.append(fairnessMetrics.calculateFairnessStats(hypothesis_indx, filtered_items, filtered_case1_pl,
                                                            filtered_case1_tl))
        case2_inequalities.append(fairnessMetrics.calculateFairnessStats(hypothesis_indx, filtered_items, filtered_case2_pl,
                                                   filtered_case2_tl))
    if usegini:
        # Use gini
        ginis = [gini(case1_inequalities), gini(case2_inequalities)]
        x = np.exp(ginis)
    else:
        # Use generalized_entropies
        generalized_entropies = [generalized_entropy(case1_inequalities), generalized_entropy(case2_inequalities)]
        x = np.exp(generalized_entropies)
    softmax = x/np.sum(x)
    #print("These should sum to 1: "+str(softmax))
    #p = sigmoid(gini1-gini2)
    #Highter the inequality, lower the probability
    #if len(test)==4:
    #    assert test[3][hypothesis_indx]==softmax[1]
    return (softmax[1], softmax[0])


def priorsAll(demographics_indices, dem_values, test):
    answer = []
    for hypothesis_indx in range(0,max_h_index):
        answer.append(priors(demographics_indices, dem_values, test, hypothesis_indx))
    #print("priorsAll",test,answer)
    return answer


#Sumx [P(x | past observations) * [Sumi P(fair_i | observations, test=x)^2]] - [Sumi P(fair_i | observations)^2]
# Delta( t | past_obs) = Sum(x, P(X_t = x | past_obs) * Sum(hi, P(hi | past_obs, X_t = x)^2)) - Sum(hi, P(hi | past_obs)^2)
#                      = term_with_test - term_without_test
def test_objective(demographics_indices, dem_values, test_observations, test, debug=False, old=False):
    #cdef double term_with_test, term_without_test, sum_i
    #cdef int x, i
    try:
        assert len(test)==4
    except:
        print("ERROR: test with incorrect length %d"%(len(test)))
        print(test)
        exit(0)
    term_without_test = 0.0
    p_obs = p_observations(demographics_indices, dem_values, test_observations)
    p_h_obs = []
    sum_p_h_obs = 0
    #HYPOTHESIS RANGE
    for i in range(0,max_h_index):
        p_h_obs.append(p_h_observations(demographics_indices, dem_values, test_observations, i, p_obs))
        term_without_test = term_without_test + math.pow(p_h_obs[i], 2)
        sum_p_h_obs += p_h_obs[i]
    
    sqsum_i = [0.0]*2
    sum_i = [0.0]*2
    for i in range(0,max_h_index):
        p_h_observations_with_test0 = p_h_observations_with_test(demographics_indices, dem_values, test_observations, i, test, 0, p_obs)
        p_h_observations_with_test1 = p_h_observations_with_test(demographics_indices, dem_values, test_observations, i, test, 1, p_obs)
        sqsum_i[0] += math.pow(p_h_observations_with_test0, 2)
        sqsum_i[1] += math.pow(p_h_observations_with_test1, 2)
        sum_i[0] += p_h_observations_with_test0
        sum_i[1] += p_h_observations_with_test1
    p_out = p_outcome(demographics_indices, dem_values, test_observations, test, [0,1], p_h_obs)
    term_with_test = sqsum_i[0]*p_out[0] + sqsum_i[1]*p_out[1]

    # check that proabilities look okay
    try:
        assert abs(1-sum_p_h_obs)<1E-5
    except:
        print(sum_p_h_obs)
        exit(0)
    try:
        assert abs(1-p_out[0]-p_out[1])<1E-5
    except:
        print(p_out[0],p_out[1],p_out[0]+p_out[1])
        exit(0)
    try:
        assert abs(1-sum_i[0])<1E-5
    except:
        print(sum_i[0])
        exit(0)
    try:
        assert abs(1-sum_i[1])<1E-5
    except:
        print(sum_i[1])
        exit(0)
    return term_with_test - term_without_test
