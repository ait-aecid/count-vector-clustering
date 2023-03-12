import math
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="data/hdfs_wuyifan18/", help="path to input files", type=str, choices=['data/hdfs_wuyifan18/', 'data/hdfs_loglizer/'])
parser.add_argument("--threshold", default=0.14, help="similarity threshold used for clustering", type=float) 
parser.add_argument("--normalize", default="True", help="if True, count vectors are normalized before clustering", type=str, choices=['True', 'False'])
parser.add_argument("--idf", default="False", help="if True, event types are weighted higher if they occur in fewer sequences", type=str, choices=['True', 'False'])

params = vars(parser.parse_args())
data_dir = params["data_dir"]
threshold = params["threshold"]
normalize = params["normalize"] == 'True'
idf = params["idf"] == 'True'

train_vectors = []
known_event_types = set()
idf_weights = {}
N = 0

# Learn all unique count vectors from training file
with open(data_dir + 'hdfs_train') as f:
    cnt = 0
    for line in f:
        cnt += 1
        # Split sequences into single event types
        parts = line.strip('\n').strip(' ').split(' ')
        train_vector = {}
        for part in parts:
            # Learn all known event types that occur in the training data set
            known_event_types.add(part)
            # Create an event count vector for each sequence
            if part in train_vector:
                train_vector[part] += 1
            else:
                train_vector[part] = 1
            if idf:
                # Count the sequences where each event type occurs in at least once
                if part in idf_weights:
                    idf_weights[part].add(cnt)
                else:
                    idf_weights[part] = set([cnt])
        if train_vector not in train_vectors:
            train_vectors.append(train_vector)
    # N stores the total number of sequences
    N = cnt
    for event_type in idf_weights:
        idf_weights[event_type] = math.log10((1 + N) / len(idf_weights[event_type]))

#print(train_vectors)

def detect_anomalies(line):
    # Returns True if anomaly is detected and False otherwise
    parts = line.strip('\n').strip(' ').split(' ')
    # Immediately detect an anomaly if sequence involves an event type that was not seen during training
    for part in parts:
        if part not in known_event_types:
            return True
    test_vector = {}
    # Create an event count vector for the currently processed sequence
    for part in parts:
        if part in test_vector:
            test_vector[part] += 1
        else:
            test_vector[part] = 1
    min_dist = None
    for train_vector in train_vectors:
        # Iterate over all known count vectors and check if there is at least one that is similar enough to consider the currently processed sequence as normal
        manh = 0
        limit = 0
        for event_type in set(list(train_vector.keys()) + list(test_vector.keys())):
            idf_fact = 1
            if idf:
                idf_fact = idf_weights[event_type]
            norm_sum_train = 1
            norm_sum_test = 1
            # Sum up the l1 norm and the highest possible distance for normalization
            if normalize:
                norm_sum_train = sum(train_vector.values())
                norm_sum_test = sum(test_vector.values())
            if event_type not in train_vector:
                manh += test_vector[event_type] * idf_fact / norm_sum_test
                limit += test_vector[event_type] * idf_fact / norm_sum_test
            elif event_type not in test_vector:
                manh += train_vector[event_type] * idf_fact / norm_sum_train
                limit += train_vector[event_type] * idf_fact / norm_sum_train
            else:
                manh += abs(train_vector[event_type] * idf_fact / norm_sum_train - test_vector[event_type] * idf_fact / norm_sum_test)
                limit += max(train_vector[event_type] * idf_fact / norm_sum_train, test_vector[event_type] * idf_fact / norm_sum_test)
        if min_dist is None:
            # Initialize min_dist for first count vector
            min_dist = manh / limit
        else:
            # Update min_dist if a more similar count vector is found
            if manh / limit < min_dist:
                min_dist = manh / limit
        if min_dist < threshold:
            # If min_dist is below the similarity threshold, the sequence is predicted to be normal
            return False
    # No sufficiently similar count vector was found; the sequence is predicted to be anomalous
    return True

# Initialize metrics
tp = 0
fn = 0
tn = 0
fp = 0

# Run detection on abnormal data
with open(data_dir + 'hdfs_test_abnormal') as f:
    for line in f:
        if detect_anomalies(line) is True:
            tp += 1
        else:
            fn += 1

# Run detection on normal data
with open(data_dir + 'hdfs_test_normal') as f:
    for line in f:
        if detect_anomalies(line) is True:
            fp += 1
        else:
            tn += 1 

# Print results
print('Threshold=' + str(threshold))
print('TP=' + str(tp))
print('FP=' + str(fp))
print('TN=' + str(tn))
print('FN=' + str(fn))
print('TPR=R=' + str(tp / (tp + fn)))
print('FPR=' + str(fp / (fp + tn)))
print('TNR=' + str(tn / (tn + fp)))
print('P=' + str(tp / (tp + fp)))
print('F1=' + str(tp / (tp + 0.5 * (fp + fn))))
print('ACC=' + str((tp + tn) / (tp + tn + fp + fn)))
