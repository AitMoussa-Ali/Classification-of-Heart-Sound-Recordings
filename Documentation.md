# Where i'm arrived on 11/07/2026 ? 
- I created my own architecture based on an MIL approach
- I evaluated and i tried to change the model each time to see the results
- The model predicts well, but the problem is that we were having an overfitting problem
- Also, we acheived a good recall and sensitivity, but we need to improve the baseline to improve the results

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

