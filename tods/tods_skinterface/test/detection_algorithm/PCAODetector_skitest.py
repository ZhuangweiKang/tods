import numpy as np
import pandas as pd
import os
from tods.tods_skinterface.primitiveSKI.detection_algorithm.PCAODetector_skinterface import PCAODetectorSKI

# X_train = np.array([[3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]])
# X_test = np.array([[3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]])
cpath = os.path.dirname(os.path.abspath(__file__))
dataset_array = pd.read_csv(os.path.join(cpath, '../../../../datasets/NAB/realTweets/labeled_Twitter_volume_AMZN.csv')).values

trainset_rate = 0.8
trainset_len = int(trainset_rate*len(dataset_array))
X_train = dataset_array[:trainset_len, [1]]
Y_train = dataset_array[:trainset_len, [2]].astype(np.int)
X_test = dataset_array[trainset_len:, [1]]
Y_test = dataset_array[trainset_len:, [2]].astype(np.int)

transformer = PCAODetectorSKI()
transformer.fit(X_train)
prediction_labels = transformer.predict(X_test)
prediction_score = transformer.predict_score(X_test)

print("Primitive: ", transformer.primitive)
print("Prediction Labels\n", prediction_labels)
print("Prediction Score\n", prediction_score)
