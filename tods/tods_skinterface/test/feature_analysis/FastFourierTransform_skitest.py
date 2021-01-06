import numpy as np
import pandas as pd
import os
from tods.tods_skinterface.primitiveSKI.feature_analysis.FastFourierTransform_skinterface import FastFourierTransformSKI

# X_test = np.array([[3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]])
cpath = os.path.dirname(os.path.abspath(__file__))
dataset_array = pd.read_csv(os.path.join(cpath, '../../../../datasets/NAB/realTweets/labeled_Twitter_volume_AMZN.csv')).values
X_test = dataset_array[:, [1]]

transformer = FastFourierTransformSKI()
X_transform = transformer.produce(X_test)

print("Primitive:", transformer.primitive)
print("X_transform:\n", X_transform)
