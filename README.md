# rippl-AI

__rippl-AI__ is an open toolbox of several artificial intelligence (AI) approaches for sharp-wave ripple (SWR) detection. In particular, this toolbox offers multiple successful plug-and-play models from 5 AI architectures (1D-CNN, 2D-CNN, LSTM, SVM and XGBoost) that are ready to use to detect SWRs in hippocampal recordings. Moreover, there is an additional package that allows easy re-training, so that models are updated to better detect particular features of your own recordings. 

# Description

## Sharp-wave ripples

Sharp-wave ripples (SWRs) are transient fast oscillatory events (100-250Hz) of around 50ms that appear in the hippocampus, that had been associated with memory consolidation. During SWRs, sequential firing of ensembles of neurons are _replayed_, reactivating memory traces of previously encoded experiences. SWR-related interventions can influence hippocampal-dependent cognitive function, making their detection crucial to understand underlying mechanisms. However, existing SWR identification tools mostly rely on using spectral methods, which remain suboptimal.

Because of the micro-circuit properties of the hippocampus, CA1 SWRs share a common profile, consisting of a _ripple_ in the _stratum pyramidale_ (SP), and a _sharp-wave_ deflection in _stratum radiatum_ that reflects the large excitatory input that comes from CA3. Yet, SWRs can extremely differ depending on the underlying reactivated circuit. This continuous recording shows this variability:

![Example of several SWRs](https://github.com/PridaLab/rippl-AI/blob/main/figures/ripple-variability.png)

## Artificial intelligence architectures

In this project, we take advantage of supervised machine learning approaches to train different AI architectures so they can unbiasedly learn to identify signature SWR features on raw Local Field Potential (LFP) recordings. These are the explored architectures:

### Convolutional Neural Networks (CNNs)

![Convolutional Neural Networks](https://github.com/PridaLab/rippl-AI/blob/main/figures/CNNs.png)

### Long-Short Term Memory Recurrent Neural Networks (LSTM)

![Long-Short Term Memory Recurrent Neural Networks](https://github.com/PridaLab/rippl-AI/blob/main/figures/LSTM.png)

### Support Vector Machine (SVM)

![Support Vector Machine](https://github.com/PridaLab/rippl-AI/blob/main/figures/SVM.png)

### Extreme-Gradient Boosting (XGBoost)

![Extreme-Gradient Boosting](https://github.com/PridaLab/rippl-AI/blob/main/figures/XGBoost.png)



# The toolbox

This toolbox contains three main blocks: **detection**, **re-training** and **exploration**. These three packages can be used jointly or separatedly. We will proceed to describe each of their purpose and usage.

## Detection

In previous works ([Navas-Olive, Amaducci et al, 2022](https://elifesciences.org/articles/77772)), we demonstrated that using feature-based algorithms to detect electrophysiological events, such as SWRs, had several advantages:
* Performance lies within the expert range
* It is more stable and less biased than spectral methods
* It can detect a wider variety of SWRs
* It can be used as an interpretation tool
All this is available in our [cnn-ripple repository](https://github.com/PridaLab/cnn-ripple).

In this toolbox, we widen the machine learning spectrum, by offering multiple plug-and-play models, from very different AI architectures: 1D-CNN, 2D-CNN, LSTM, SVM and XGBoost. We performed an exhaustive parametric search to find different `architecture` solutions (i.e. `model`s) that achieve:
* **High performance**, so detections were as similar as manually labeled SWRs
* **High stability**, so performance does not depend on threshold selection
* **High generability**, so performance remains good on very different contexts

This respository contains the best five `model`s from each of these five `architecture`s. These `model`s are already trained using mice data, and can be found in the [optimized_models/](https://github.com/PridaLab/rippl-AI/blob/main/optimized_models/) folder. 

The [rippl_AI](https://github.com/PridaLab/rippl-AI/blob/main/rippl_AI.py) python module contains all the necessary functions to easily use any `model` to detect SWRs. Additionally, we also provide some auxiliary functions in the [aux](https://github.com/PridaLab/rippl-AI/blob/main/aux.py) module, that contains useful code to process LFP and evaluate performance detection.

Moreover, several usage examples of all functions can be found in the [examples_detection/](https://github.com/PridaLab/rippl-AI/blob/main/examples_detection/) folder.



### rippl_AI.predict()

The python function `predict()` of the `rippl_AI` module computes the SWR probability for a give LFP. 

In the figure below, you can see an example of a high-density LFP recording (top) with manually labeled data (gray). The objective of these `model`s is to generate an output signal that most similarly matches the manually labeled signal. The output of the uploaded optimized models can be seen in the bottom, where outputs go from 0 (low probability of SWR) to 1 (high probability of SWR) for each LFP sample.

![Detection method](https://github.com/PridaLab/rippl-AI/blob/main/figures/output-probabilities.png)

The `rippl_AI.predict()` input and output variables are:

* Mandatory inputs: 
	- `LFP`: LFP recorded data (`np.array`: `n_samples` x `n_channels`). Although there are no restrictions in `n_channels`, some considerations should be taken into account (see `channels`). Data does not need to be normalized, because it will be internally be z-scored (see `aux.process_LFP()`). 
	- `sf`: sampling frequency (in Hz).

* Optional inputs:
	- `arch`: 
	- `model_number`: 
	- `channels`: Channels to be used for detection (`np.array` or `list`: `1` x `8`). This is the most senstive parameter, because models will be looking for specific spatial features over all channels. The two main remarks are:
		* All models have been trained to look at features in the pyramidal layer (SP), so for them to work at their maximum potential, the selected channels would ideally be centered in the SP, with a postive deflection on the first channels (upper channels) and a negative deflection on the last channels (lower channels). The image above can be used as a visual reference of how to choose channels.
		* For all combinations of `architectures` and `model_numbers`, `channels` **has to be of size 8**. There is only one exception, for `architecture = 2D-CNN` with `models = {3, 4, 5}`, that needs to have **3 channels**. 
		* If you are using a high-density probe, then we recommend to use equi-distant channels from the beginning to the end of the SP. For example, for Neuropixels in mice, a good set of channels would be `pyr_channel` + [-8,-6,-4,-2,0,2,4,6]. 
		* In the case of linear probes or tetrodes, there are not enough density to cover the SP with 8 channels. For that, interpolation or recorded channels can be done without compromising performance. New artificial interpolated channels will be add to the LFP wherever there is a `-1` in `channels`. For example, if `pyr_channel=11` in your linear probe, so that 10 is in _stratum oriens_ and 12 in _stratum radiatum_, then we could define `channels=[10,-1,-1,11,-1,-1,-1,12]`, where 2nd and 3rd channels will be an interpolation of SO and SP channels, and 5th to 7th an interpolation of SP and SR channels. For tetrodes, organising channels according to their spatial profile is very convenient to assure best performance. These interpolations are done using the function `aux.interpolate_channels()`.
		* Several examples of all these usages can be found in the [examples_detection/](https://github.com/PridaLab/rippl-AI/blob/main/examples_detection/) folder.

* Output:
	- `SWR_prob`: model output for every sample of the LFP (`np.array`: `n_samples` x 1). It can be interpreted as the confidence or probability of a SWR event, so values close to 0 mean that the `model` is certain that there are not SWRs, and values close to 1 that the model is very sure that there is a SWR hapenning.
	- `LFP_norm`: LFP data used as an input to the model (`np.array`: `n_samples` x `len(channels)`). It is undersampled to 1250Hz, z-scored, and transformed to used the channels specified in `channels`.



### rippl_AI.get_intervals()

The python function `get_intervals()` of the `rippl_AI` module takes the output of `rippl_AI.predict()` (i.e. the SWR probability), and identifies SWR beginnings and ends by stablishing a threshold. In the figure below, you can see how the threshold can decisevely determine what events are being detected. For example, lowering the threshold to 0.5 would have result in XGBoost correctly detecting the first SWR, and the 1D-CNN detecting the sharp-wave that has no ripple.

![Detection method](https://github.com/PridaLab/rippl-AI/blob/main/figures/detection-method.png)

* Mandatory inputs: 
	- `SWR_prob`: output of `rippl_AI.predict()`. If this is the only input, the function will display a histogram of all SWR probability values (i.e. `n_samples`), and a draggable threshold to set a threshold based on the values of this particular session. When 'Done' button is pressed, the GUI takes the value of the draggable as the threshold, and computes the beginning and ends of the events.

* Optional inputs - **Setting the threshold**
	Depending on the inputs, different possibilities arise:
    - `threshold`: Threshold of predictions (`float`)
    - `LFP_norm`: Normalized input signal of the model (`np.array`: `n_samples` x `n_channels`). It is recommended to use `LFP_norm`.
    - `file_path`: Absolute path of the folder where the .txt with the predictions will be generated (`string`). Leave empty if you don't want to generate the file. 
    - `win_size`: Length of the displayed ripples in miliseconds (`integer`). By default 100 ms.
    - `sf`: Sampling frequency (Hz) of `LFP_norm` (`integer`). By default 1250 Hz (i.e., sampling frequency of `LFP_norm`).

    There are 4 possible use cases, depending on which parameter combination is used when calling the function.
    1. `rippl_AI.get_intervals(SWR_prob)`: a histogram of the output is displayed, you drag a vertical bar to selecct your `threshold`
    2. `rippl_AI.get_intervals(SWR_prob,threshold)`: no GUI is displayed, the predictions are gererated automatically
    3. `rippl_AI.get_intervals(SWR_prob,LFP_norm)`: some examples of detected events are displayed next to the histogram
    4. `rippl_AI.get_intervals(SWR_prob,LFP_norm,threshold)`: same case as 3, but the initial location of the bar is `threshold`

    Examples:
	- `get_intervals(SWR_prob, LFP_norm=LFP_norm, sf=sf, win_size=win_size)`: as `LFP_norm` is also added as an input, then the GUI adds up to 50 examples of SWR detections. If the 'Update' button is pressed, another 50 random detections are shown. When 'Save' button is pressed, the GUI takes the value of the draggable as the threshold. Sampling frequency `sf` (in Hz) and window size `win_size` (in milliseconds) can be used to set the window length of the displayed examples. It automatically discards false positives due to drifts, but if you want to set it off, you can set `discard_drift` to `false`. By default, it discards noises whose mean LFP is above `std_discard` times the standard deviation, which by default is 1SD. This parameter can also be changed. ![Detection method](https://github.com/PridaLab/rippl-AI/blob/main/figures/threshold-selection.png)
	- `get_intervals(SWR_prob, 'threshold', threshold)`: if a threshold is given, then it takes that threshold without displaying any GUI.

* Outputs:
	- `predictions`: Returns the time (in seconds) of the begining and end of each vents. (`n_events` x 2)



### aux.process_LFP()

The python function `process_LFP()` of the `aux` module processes the LFP before it is input to the algorithm. It downsamples LFP to 1250 Hz, and normalizes each channel separately by z-scoring them.

* Mandatory inputs:

* Optional inputs:

* Output:


### aux.interpolate_channels()

The python function `interpolate_channels()` of the `aux` module allows creating more intermediate channels using interpolation. 

Because these models best performed using a richer spatial profile, they need 8 channels as an input. However, it is possible that some times we cannot get such number of channels in the pyramidal layer, like when using linear probes (only 2 oe 3 channels fit in the pyramidal layer) or tetrodes (there are 4 recording channels).

For this, we developed this interpolation function, that creates new channels between any pair of your recording channels. Using this approach, we can successfully use the already built algorithms with an equally high performance.

* Mandatory inputs:

* Optional inputs:

* Output:

Usage examples can be found in the [examples_detection/](https://github.com/PridaLab/rippl-AI/blob/main/examples_detection/) folder.


### aux.get_performance()

The python function `get_performance()` of the `aux` module computes several performance metrics:
* precision: also called *positive predictive value* is computed as (# good detections) / (# all detections)
* recall: also called *sensitivity* is computed as (# good detections) / (# all ground truth events)
* F1: computed as the harmonic mean between precision and recall, is a conservative and fair measure of performance. If any of precision or recall is low, F1 will be low. F1=1 only happens if detected events exactly match ground truth events.

Therefore, this function can be used only when some ground truth is given.

* Mandatory inputs:

* Optional inputs:

* Output:




## Re-training




## Exploration





# Enviroment setup

On Windows 10:
  Using the enviroment file 'env_PublicBCG_d.yml':
1. Install miniconda, following the tutorial: https://docs.conda.io/en/latest/miniconda.html
2. Launch the anaconda console, typing anaconda promp in the windows search bar.
3. Navigate to the project directory, using 'cd' to change folders, for example:
  cd C:\Myprojects\Public_BCG_models
4. Type in the anaconda prompt: 
   conda env create -f rippl_AI_env.yml
5. This will create a enviroment in your miniconda3 enviroments folder, usually:
  C:\Users\<your_user>\miniconda3\envs
6. Check with:
  conda env list
  The enviroment 'rippl_AI_env' should be included in the list
7. Activate the enviroment with 'conda activate rippl_AI_env', in case you want to launch the scripts from the command prompt. If you are using Visual Studio Code, you need to select the python interpreter 'conda activate rippl_AI_env'

If generating the enviroment from the .yml file fails, you can manually install the required packages, creating a new conda enviroment and following the next steps:
1. conda create -n rippl_AI_env python=3.9.15
1. conda install pip
2. pip install numpy
3. pip install tensorflow
4. pip install keras
5. pip install xgboost
6. pip install -U scikit-learn  (on ubuntu)
7. pip install imblearn
8. pip install matplotlib

If you want to download the lab data from figshare (not normalized, sampled with the original frequency of 30 000 Hz):
1. git clone https://github.com/cognoma/figshare.git
2. cd figshare
2. python setup.py

If you want to check how the data preprocessing works or implement your own:
1. pip install pandas
  
  https://www.cs.toronto.edu/~duvenaud/distill_bayes_net/public/
