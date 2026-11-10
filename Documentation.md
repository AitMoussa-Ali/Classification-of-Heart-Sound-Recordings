# Where i'm arrived on 11/07/2026 ? 
- I created my own architecture based on an MIL approach
- I evaluated and i tried to change the model each time to see the results
- The model predicts well, but the problem is that we were having an overfitting problem
- Also, we acheived a good recall and sensitivity, but we need to improve the baseline to improve the results

# What are the problems with time series data type ? 
**Review of my baseline** :

- First in my baseline, we are using a cnn for feature extraction, each segment of signal will produce a vector of 20 features, at the end we feed each vector to an MLP to predict a logit.
- We repeat this process to all the segments of the signal, at the end we take the mean probability as an MIL output so we can classify the entire signal.

**Problems**:
- The architecture is not really relevant, because i'm not doing an agreagation for the extracted features. 
- We are dealing with time series data, so the thing is that the order of the series may introduce more information for feature extraction and classification.
So what we need to do is to figure out a way to add information about the positions of each segment, so we can introduce more information for the classification purpose.


# What my professor suggusted to include to improve the results ? 
- Include an attention mechanism instead of an agregation to classify the signals
- Explore CNN architecture but with different kernel size to capture more complex patterns
- Try to study how can we integrate a mecanisme that can take into consideration tthe times series relationships between different segments, instead of a simple agregation like a mean or max


# What my professor suggusted to read ? 
- TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning : based on time series analysis using multiple instance learning
- Detection of Coronary Artery Disease Based on Clinical Phonocardiogram and Multiscale Attention Convolutional Compression Network (IEEE JBHI, 2024), pour l'exploitation de convolutions multi-échelles et de l'attention sur les signaux PCG.
- Heart sound classification based on convolutional neural network with convolutional block attention module (Frontiers in Physiology, 2025), pour l'intégration de mécanismes d'attention dans les architectures CNN appliquées aux signaux cardiaques.

# Summary of the lectures : 
## **TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning : based on time series analysis using multiple instance learning** : 

The TimeMIL is an architecture based on the strandard MIL. 

MIL uses a CNN as a feature extractor for each segment of the signal, at the end we are going to have a vector of 128 features by applying a max pooling or mean pooling. After features extraction, we feed the vector of features into an MLP to classify the entire signal, and not each segment. 

TimeMIL proposed another architecture, and that by replacing the pooling phase by an attention layer, add a cls token to capture all the seqeunce, and add a positional encoding to encode the position of each segment so it can help the attention layer to capture the temporal features and relations.

The positional encoding they used in this approach, is not like the positional encoding used in LLM's, the key difference is that : 

**- Positional encoding in classical transformers** is predefined and not learnable, it's generated out of the input context, so it can't capture both time and frequency information.
**- Wavelet positional encoding** is a learnable positional encoding with a conditional positional encoding

