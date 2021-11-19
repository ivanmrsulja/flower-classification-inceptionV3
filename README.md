# flower-classification-inceptionV3
Flower classification model that classifies flowers in 10 classes.

- Training is done using a pre-anotated dataset from Kaggle (https://www.kaggle.com/olgabelitskaya/flower-color-images).
- Dataset is split into training and validation sub-sets (80-10).
- Model is made using transfer learning with an InceptionV3 model with one added fully-connected layer (1024 neurons) and a softmax exit layer.

Dataset is very small but data is evenly distributed so accuracy can be used as a valuable metric. In my implementation accuracy is about 85% while F1 value is about 60%. This does not seem like a lot but this can be greatly improved with a larger dataset and a longer training time.
