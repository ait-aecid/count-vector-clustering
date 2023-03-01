# count-vector-clustering

The algorithm in this repository evaluates the semi-supervised count-vector-clustering approach described in [1] on the HDFS log data set [2]. In short, the approach creates count vectors for each event sequence in the training data set and predicts counts vectors of new sequences from the test data set as anomalous when they are not similar enough to any of the sequences known to be normal, where the similarity metric is based on the l1-norm. In addition, all new sequences that contain event types not seen during training are predicted to be anomalous.

Run the algorithm using the following command:

```
ubuntu@user-1:~/stide$ python3 count_vector_clustering.py
Threshold=0.14
TP=16826
FP=376
TN=552990
FN=12
TPR=R=0.9992873262857822
FPR=0.000679477958530159
TNR=0.9993205220414698
P=0.9781420765027322
F1=0.9886016451233842
ACC=0.9993195417780303
```

The algorithm achieves an F1-Score of 98.86\% on the HDFS data set. This exceeds the detection performance achieved by [n-gram-based detection](https://github.com/ait-aecid/stide) (F1-Score is 95.14\%) as well as many deep learning approaches.

You can specify the threshold for clustering as well as changing normalization and weighting at the beginning of the python script. When count vectors are not normalized, we achieved an F1-Score of 98.85\% using a threshold of 0.11. IDF weighting does not seem to improve the results any further.

[1] Landauer M., Skopik F., Höld G., Wurzenberger M. (2022): [A User and Entity Behavior Analytics Log Data Set for Anomaly Detection in Cloud Computing.](https://doi.org/10.1109/BigData55660.2022.10020672) 2022 IEEE International Conference on Big Data - 6th International Workshop on Big Data Analytics for Cyber Intelligence and Defense (BDA4CID 2022), December 17-20, 2022, Osaka, Japan. IEEE. \[[PDF](https://www.skopik.at/ait/2022_bigdata.pdf)\]

[2] HDFS log data set taken without changes from the [DeepLog implementation by wuyifan18](https://github.com/wuyifan18/DeepLog)
