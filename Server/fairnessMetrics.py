" " " helper functions for calculating fairness metrics: " \
"1. False Positive, False Negative, " \
"2. False Omission, False Discovery, " \
"3. Demographic Parity, " \
"4. Accuracy Parity, " \
"5.Individual" \
"" " "
import numpy as np

error_value = -1
fairness_theories = ["Accuracy", "False Discovery", "False Negative", "Demographic Parity", "Individual", "False Omission", "False Positive", "True Positive"]


def calculateFairnessStats(stats_index, features, predicted, true_labels):
    if stats_index==0:
        return accuracyRate(features, predicted, true_labels)
    elif stats_index==1:
        return falseDiscoveryRate(features, predicted, true_labels)
    elif stats_index==2:
        return falseNegativeRate(features, predicted, true_labels)
    elif stats_index==3:
        return demographicParity(features, predicted, true_labels)
    elif stats_index==4:
        return individualFairness(features, predicted, true_labels)
    elif stats_index==5:
        return falseOmissionRate(features, predicted, true_labels)
    elif stats_index==6:
        return falsePositiveRate(features, predicted, true_labels)
    elif stats_index==7:
        return truePositiveRate(features, predicted, true_labels)
    else:
        return "Invalid index"


def demographicParity(features, predicted_labels, true_labels):
    if (not ((len(features) == len(predicted_labels)) and (len(predicted_labels) == len(true_labels)))):
        print("Length of inputs unequal")
        return error_value
    num_predicted_postives = 0
    for i in range(len(predicted_labels)):
        if (predicted_labels[i] == 1):
            num_predicted_postives = num_predicted_postives + 1
    return num_predicted_postives / len(predicted_labels)


def falsePositiveRate(features, predicted_labels, true_labels):
    #if(not ((len(features) == len(predicted_labels)) and (len(predicted_labels) == len(true_labels)) )):
    #    print("Length of inputs unequal")
    #    return error_value
    #cdef int num_false_positives, num_negatives, i
    num_false_positives = 0
    num_negatives = 0
    for i in range(len(predicted_labels)):
        if (true_labels[i] == 0):
            num_negatives += 1
            if (predicted_labels[i] == 1):
                num_false_positives += 1
    if (num_negatives == 0):
        return 5
    return num_false_positives / num_negatives


def falseNegativeRate(features, predicted_labels, true_labels):
    #if(not ((len(features) == len(predicted_labels)) and (len(predicted_labels) == len(true_labels)) )):
    #    print("Length of inputs unequal")
    #    return error_value
    #cdef int num_false_negatives, num_positives, i
    num_false_negatives = 0
    num_positives = 0
    for i in range(len(predicted_labels)):
        if (true_labels[i] == 1):
            num_positives += 1
            if (predicted_labels[i] == 0):
                num_false_negatives += 1
    if (num_positives == 0):
        return 5
    return num_false_negatives / num_positives


def truePositiveRate(features, predicted_labels, true_labels):
    return 1 - falseNegativeRate(features, predicted_labels, true_labels)


def trueNegativeRate(features, predicted_labels, true_labels):
    return 1 - falsePositiveRate(features, predicted_labels, true_labels)


def falseOmissionRate(features, predicted_labels, true_labels):
    #if(not ((len(features) == len(predicted_labels)) and (len(predicted_labels) == len(true_labels)) )):
    #    print("Length of inputs unequal")
    #    return error_value
    #cdef int num_false_negatives, num_predicted_negatives
    num_false_negatives = 0
    num_predicted_negatives = 0
    for i in range(len(predicted_labels)):
        if (predicted_labels[i] == 0):
            num_predicted_negatives += 1
            if (true_labels[i] == 1):
                num_false_negatives += 1
    if (num_predicted_negatives == 0):
        return 5
    return num_false_negatives / num_predicted_negatives


def falseDiscoveryRate(features, predicted_labels, true_labels):
    #if(not ((len(features) == len(predicted_labels)) and (len(predicted_labels) == len(true_labels)) )):
    #    print("Length of inputs unequal")
    #    return error_value
    #cdef int num_false_positives, num_predicted_positive, i
    num_false_positives = 0
    num_predicted_positive = 0
    for i in range(len(predicted_labels)):
        if (predicted_labels[i] == 1):
            num_predicted_positive += 1
            if (true_labels[i] == 0):
                num_false_positives += 1
    if (num_predicted_positive == 0):
        return 5
    return num_false_positives / num_predicted_positive


def accuracyRate(features, predicted_labels, true_labels):
    #if(not ((len(features) == len(predicted_labels)) and (len(predicted_labels) == len(true_labels)) )):
    #    print("Length of inputs unequal")
    #    return error_value
    #cdef int num_misclassified, i
    num_misclassified = 0
    for i in range(len(predicted_labels)):
        if (predicted_labels[i] != true_labels[i]):
            num_misclassified += 1
    return 1.0 - num_misclassified / len(predicted_labels)


def individualFairness(features, predicted_labels, true_labels):
    #if (not ((len(features) == len(predicted_labels)) and (len(predicted_labels) == len(true_labels)))):
    #    print("Length of inputs unequal")
    #    return error_value
    individual_benefits = []
    for i in range(len(predicted_labels)):
        benefit = (predicted_labels[i] - true_labels[i]) + 1
        individual_benefits.append(benefit)
    return individual_benefits, np.mean(np.array(individual_benefits))