import numpy as np
from tods.tods_skinterface.primitiveSKI.detection_algorithm.SystemWiseDetection_skinterface import SystemWiseDetectionSKI

X_train = np.asarray([[1,2,3,4,5,6,7,8,9,10,11,12],[1,1,1,1,5,5,5,5,3,3,3,3],[1.0,4.0,5.0,6.0,2.0,1.0,9.0,10.0,3.0,4.0,18.0,1.0]])

transformer = SystemWiseDetectionSKI()

prediction_labels = transformer.produce(X_train)

print("Primitive: ", transformer.primitive)
print("Prediction Labels\n", prediction_labels)
#print("Prediction Score\n", prediction_score)
