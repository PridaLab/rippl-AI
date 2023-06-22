# rippl-AI

# Description

__rippl-AI__ is an open toolbox of several artificial intelligence (AI) approaches for sharp-wave ripple (SWR) detection. In particular, this toolbox offers multiple successful plug-and-play models from 5 AI architectures (1D-CNN, 2D-CNN, LSTM, SVM and XGBoost) that are ready to use to detect SWRs in hippocampal recordings. Moreover, there is an additional package that allows easy re-training, so that models are updated to better detect particular features of your own recordings. 

## Sharp-wave ripples

Sharp-wave ripples (SWRs) are transient fast oscillatory events (100-250Hz) of around 50ms that appear in the hippocampus, that had been associated with memory consolidation. During SWRs, sequential firing of ensembles of neurons are _replayed_, reactivating memory traces of previously encoded experiences. SWR-related interventions can influence hippocampal-dependent cognitive function, making their detection crucial to understand underlying mechanisms. However, existing SWR identification tools mostly rely on using spectral methods, which remain suboptimal.

Because of the micro-circuit properties of the hippocampus, CA1 SWRs share a common profile, consisting of a _ripple_ in the _stratum pyramidale_, and a _sharp-wave_ deflection in _stratum radiatum_ that reflects the large excitatory input that comes from CA3. Yet, SWRs can extremely differ depending on the underlying reactivated circuit. This continuous recording shows this variability:

![Example of several SWRs](https://github.com/PridaLab/rippl-AI/blob/main/figures/ripple-variability.png)

## Artificial intelligence architectures

In this project, we take advantage of supervised machine learning approaches to train different AI architectures so they can unbiasedly learn to identify signature SWR features on raw Local Field Potential (LFP) recordings. These are the explored architectures:

### Convolutional Neural Networks (CNNs)

![Convolutional Neural Networks](https://github.com/PridaLab/rippl-AI/blob/main/figures/CNNs.png)

### Long-Short Term Memory Recurrent Neural Networks (LSTM)

![Long-Short Term Memory Recurrent Neural Networks](https://github.com/PridaLab/rippl-AI/blob/main/figures/LSTM.png)

### Extreme-Gradient Boosting (XGBoost)

![Extreme-Gradient Boosting](https://github.com/PridaLab/rippl-AI/blob/main/figures/XGBoost.png)

### Support Vector Machine (SVM)

![Support Vector Machine](https://github.com/PridaLab/rippl-AI/blob/main/figures/SVM.png)


