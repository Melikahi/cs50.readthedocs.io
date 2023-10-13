import csv
import sys
import calendar
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=TEST_SIZE)

    # Train model and make predictions
    predictions = []

    for test_sample in X_test:
        nearest_neighbor = find_nearest_neighbor(test_sample, X_train, y_train)
        predictions.append(nearest_neighbor)

    sensitivity, specificity = evaluate(y_test, predictions)
    f1_score = calculate_f1_score(y_test, predictions)

    # Calculate the number of correct predictions
    correct_predictions = np.sum(np.array(y_test) == np.array(predictions))
    incorrect_predictions = np.sum(np.array(y_test) != np.array(predictions))

    # Print results
    print(f"Correct: {correct_predictions}")
    print(f"Incorrect: {incorrect_predictions}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    months = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')

    evidence = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)


def find_nearest_neighbor(test_sample, X_train, y_train):
    # Initialize variables for nearest neighbor search
    nearest_distance = float('inf')
    nearest_neighbor = None

    for train_sample, label in zip(X_train, y_train):
        distance = euclidean_distance(test_sample, train_sample)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_neighbor = label

    return nearest_neighbor

def euclidean_distance(sample1, sample2):
    # Calculate Euclidean distance between two samples
    distance = 0.0
    for i in range(len(sample1)):
        distance += (sample1[i] - sample2[i]) ** 2
    return math.sqrt(distance)

def calculate_f1_score(y_true, y_pred):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    false_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    false_negatives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model

def evaluate(labels, predictions):
    sensitivity = float(0)
    specificity = float(0)

    total_positive = float(0)
    total_negative = float(0)

    for label, prediction in zip(labels, predictions):

        if label == 1:
            total_positive += 1
            if label == prediction:
                sensitivity += 1

        if label == 0:
            total_negative += 1
            if label == prediction:
                specificity += 1

    sensitivity /= total_positive
    specificity /= total_negative

    return sensitivity, specificity

if __name__ == "__main__":
    main()
