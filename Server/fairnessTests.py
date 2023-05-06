#functions to generate tests - a test is a tuple of dictionaries
from collections import defaultdict
from itertools import combinations_with_replacement
import itertools
from sympy.utilities.iterables import multiset_permutations
import fairnessMetrics
import numpy as np
from heapq import *
import random
from probabilities import priorsAll
import copy

import numpy as np
try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

rgstate_test = None


def check_valid_confusion(list1, list2):
    if (len(list1) != len(list2)):
        print("Can't compare unequal lists")
        return False
    tp = False
    fp = False
    tn = False
    fn = False
    for idx in range(0, len(list1)):
        if (list1[idx] == 0 and list2[idx] == 1):
            fp=True
        if (list1[idx] == 0 and list2[idx] == 0):
            tp = True
        if (list1[idx] == 1 and list2[idx] == 1):
            tn = True
        if (list1[idx] == 1 and list2[idx] == 0):
            fn = True
    return (tp and fp and tn and fn)


def TestIsUseful(test,margin=0.05):
    prior_probs = test[3]
    for p1, p0 in prior_probs:
        if (abs(p1-p0)>margin):
            return True
    return False

def PruneTestQueue1(tq):
    return [t for t in tq if TestIsUseful(t)]


def fill_test_queue(items_truelabel_list, demographics_indices, dem_values, possible_labels=[0,1], rgseed=None):
    rgstate_saved = random.getstate()
    random.seed(rgseed)
    S = set()
    testQueue = []
    MinusInfinity = -1.0*float("inf")
    combinations_list = list(combinations_with_replacement(possible_labels, items_truelabel_list[0][0].shape[0]))
    print("Building Test Queue")
    for combination in combinations_list:
        permutations = list(multiset_permutations(combination))
        for index1, label_permutation1 in enumerate(permutations):
            for index2, label_permutation2 in enumerate(permutations[index1:]):
                if (label_permutation1 != label_permutation2 and check_valid_confusion(label_permutation1, label_permutation2)):
                    item_idx = random.randint(0, len(items_truelabel_list)-1)
                    for items_truelabel in  [items_truelabel_list[item_idx]]:
                        if (not (check_valid_confusion(label_permutation1, items_truelabel[1]) and check_valid_confusion(label_permutation2, items_truelabel[1]))):
                            continue
                        acc1 = fairnessMetrics.accuracyRate(items_truelabel[0].tolist(), label_permutation1, items_truelabel[1])
                        acc2 = fairnessMetrics.accuracyRate(items_truelabel[0].tolist(), label_permutation2, items_truelabel[1])
                        if (acc1 == acc2 and acc1 > .5):
                            case1 = defaultdict(list)
                            case2 = defaultdict(list)
                            case1["pl"] = label_permutation1
                            case2["pl"] = label_permutation2
                            case1["tl"] = items_truelabel[1]
                            case2["tl"] = items_truelabel[1]
                            case1["item_idx"] = item_idx 
                            case2["item_idx"] = item_idx
                            case_tuple = (case1, case2)
                            prior_probs = priorsAll(demographics_indices, dem_values, (MinusInfinity, None, case_tuple))
                            S.add(str(prior_probs))
                            #heappush(testQueue, (MinusInfinity, z, case_tuple, prior_probs))
                            testQueue.append((MinusInfinity, None, case_tuple, prior_probs))
                            #heappush(testQueue, (-1.0*float("inf"), id(case_tuple), case_tuple))
        print("CURRENT COMBINATION: "+str(combination) + " Queue Size: "+str(len(testQueue)))
    testQueue = PruneTestQueue1(testQueue)
    print("Queue Size after pruning: "+str(len(testQueue)))
    random.shuffle(testQueue)
    testQueueShuffled = [(testQueue[i][0], i, testQueue[i][2], testQueue[i][3]) for i in range(len(testQueue))]
    heapify(testQueueShuffled)
    random.setstate(rgstate_saved)
    return testQueueShuffled


def random_items_generator(num_items, num_features):
    items = np.random.randint(2, size=(num_items, num_features))
    return items
