# **What should i test before moving to attention architecture**

- First thing to test is to add an agregation function like max pooling or mean pooling after the features extraction, and implement a signal label prediction instead of segment level prediction to predict the signal class at the end. And then change the feature extractor to a larger dimension, like to 128 feature vector (embedding).

- Second thing is to try to enhance the architecture, and that by reading the scientific papers.

