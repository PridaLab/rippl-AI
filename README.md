# rippl-AI

<img src="https://github.com/PridaLab/rippl-AI/blob/main/figures/rippl-AI-logo.png" width="200">

__rippl-AI__ is an open toolbox of Artifical Intelligence (AI) resources for detection of hippocampal neurophysiological signals, in particular sharp-wave ripples (SWR). This toolbox offers multiple successful plug-and-play machine learning (ML) models from 5 different architectures (1D-CNN, 2D-CNN, LSTM, SVM and XGBoost) that are ready to use to detect SWRs in hippocampal recordings. Moreover, there is an additional package that allows easy re-training, so that models are updated to better detect particular features of your own recordings. More details in [Navas-Olive, Rubio, et al. Commun Biol 7, 211 (2024)](https://www.nature.com/articles/s42003-024-05871-w)!

# Description

## Sharp-wave ripples

Sharp-wave ripples (SWRs) are transient fast oscillatory events (100-250Hz) of around 50ms that appear in the hippocampus, that had been associated with memory consolidation. During SWRs, sequential firing of ensembles of neurons are _replayed_, reactivating memory traces of previously encoded experiences. SWR-related interventions can influence hippocampal-dependent cognitive function, making their detection crucial to understand underlying mechanisms. However, existing SWR identification tools mostly rely on using spectral methods, which remain suboptimal.

Because of the micro-circuit properties of the hippocampus, CA1 SWRs share a common profile, consisting of a _ripple_ in the _stratum pyramidale_ (SP), and a _sharp-wave_ deflection in _stratum radiatum_ that reflects the large excitatory input that comes from CA3. Yet, SWRs can extremely differ depending on the underlying reactivated circuit. This continuous recording shows this variability:

![Example of several SWRs](https://github.com/PridaLab/rippl-AI/blob/main/figures/ripple-variability.png)

## Artificial intelligence architectures

In this project, we take advantage of supervised machine learning approaches to train different AI architectures so they can unbiasedly learn to identify signature SWR features on raw Local Field Potential (LFP) recordings. These are the explored architectures:

### Convolutional Neural Networks (CNNs)

![Convolutional Neural Networks](https://github.com/PridaLab/rippl-AI/blob/main/figures/CNNs.png)

### Support Vector Machine (SVM)

![Support Vector Machine](https://github.com/PridaLab/rippl-AI/blob/main/figures/SVM.png)

### Long-Short Term Memory Recurrent Neural Networks (LSTM)

![Long-Short Term Memory Recurrent Neural Networks](https://github.com/PridaLab/rippl-AI/blob/main/figures/LSTM.png)

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

The [rippl_AI](https://github.com/PridaLab/rippl-AI/blob/main/rippl_AI.py) python module contains all the necessary functions to easily use any `model` to detect SWRs. Additionally, we also provide some auxiliary functions in the [aux_fcn](https://github.com/PridaLab/rippl-AI/blob/main/aux_fcn.py) module, that contains useful code to process LFP and evaluate performance detection.

Moreover, several usage examples of all functions can be found in the [examples_detection.ipynb](https://github.com/PridaLab/rippl-AI/blob/main/examples_detection.ipynb) python notebook.



### rippl_AI.predict()

The python function `predict(LFP, sf, arch='CNN1D', model_number=1, channels=np.arange(8), d_sf=1250)` of the `rippl_AI` module computes the SWR probability for a give LFP. 

In the figure below, you can see an example of a high-density LFP recording (top) with manually labeled data (gray). The objective of these `model`s is to generate an output signal that most similarly matches the manually labeled signal. The output of the uploaded optimized models can be seen in the bottom, where outputs go from 0 (low probability of SWR) to 1 (high probability of SWR) for each LFP sample.

![Detection method](https://github.com/PridaLab/rippl-AI/blob/main/figures/output-probabilities.png)

The `rippl_AI.predict()` input and output variables are:

* Mandatory inputs: 
	- `LFP`: LFP recorded data (`np.array`: `n_samples` x `n_channels`). Although there are no restrictions in `n_channels`, some considerations should be taken into account (see `channels`). Data does not need to be normalized, because it will be internally be z-scored (see `aux_fcn.process_LFP()`). 
	- `sf`: sampling frequency (in Hz).

* Optional inputs:
	- `arch`: Name of the AI architecture to use (`string`). It can be: `CNN1D`, `CNN2D`, `LSTM`, `SVM` or `XGBOOST`.
	- `model_number`: Number of the model to use (`integer`). There are six different models for each architecture, sorted by performance, being `1` the best, and `5` the last. `model_number=6` model can be used if single-channel data needs to be used. 
	- `channels`: Channels to be used for detection (`np.array` or `list`: `1` x `8`). This is the most senstive parameter, because models will be looking for specific spatial features over all channels. Counting starts in `0`. The two main remarks are:
		* All models have been trained to look at features in the pyramidal layer (SP), so for them to work at their maximum potential, the selected channels would ideally be centered in the SP, with a postive deflection on the first channels (upper channels) and a negative deflection on the last channels (lower channels). The image above can be used as a visual reference of how to choose channels.
		* For all combinations of `architectures` and `model_numbers`, `channels` **has to be of size 8**. There is only one exception, for `architecture = 2D-CNN` with `models = {3, 4, 5}`, that needs to have **3 channels**. 
		* If you are using a high-density probe, then we recommend to use equi-distant channels from the beginning to the end of the SP. For example, for Neuropixels in mice, a good set of channels would be `pyr_channel` + [-8,-6,-4,-2,0,2,4,6]. 
		* In the case of linear probes or tetrodes, there are not enough density to cover the SP with 8 channels. For that, interpolation or recorded channels can be done without compromising performance. New artificial interpolated channels will be add to the LFP wherever there is a `-1` in `channels`. For example, if `pyr_channel=11` in your linear probe, so that 10 is in _stratum oriens_ and 12 in _stratum radiatum_, then we could define `channels=[10,-1,-1,11,-1,-1,-1,12]`, where 2nd and 3rd channels will be an interpolation of SO and SP channels, and 5th to 7th an interpolation of SP and SR channels. For tetrodes, organising channels according to their spatial profile is very convenient to assure best performance. These interpolations are done using the function `aux_fcn.interpolate_channels()`.
		* Several examples of all these usages can be found in the [examples_detection.ipynb](https://github.com/PridaLab/rippl-AI/blob/main/examples_detection.ipynb) python notebook.
	- `new_model`: Other re-trained model you want to use for detection. If you have used our re-train function to adapt the optimized models to your own data (see `rippl_AI.retrain()` for more details), you can input the `new_model` here to use that model to predict your events.
  	- `d_sf`: Desired subsampling frequency in Hz (`int`). By default all works in 1250 Hz, but can be changed if you retrain your models using `rippl_AI.retrain_model`.

* Output:
	- `SWR_prob`: model output for every sample of the LFP (`np.array`: `n_samples` x 1). It can be interpreted as the confidence or probability of a SWR event, so values close to 0 mean that the `model` is certain that there are not SWRs, and values close to 1 that the model is very sure that there is a SWR hapenning.
	- `LFP_norm`: LFP data used as an input to the model (`np.array`: `n_samples` x `len(channels)`). It is undersampled to 1250Hz, z-scored, and transformed to used the channels specified in `channels`.


### rippl_AI.get_intervals()

The python function `get_intervals(SWR_prob, LFP_norm=None, sf=1250, win_size=100, threshold=None, file_path=None, merge_win=0)` of the `rippl_AI` module takes the output of `rippl_AI.predict()` (i.e. the SWR probability), and identifies SWR beginnings and ends by stablishing a threshold. In the figure below, you can see how the threshold can decisevely determine what events are being detected. For example, lowering the threshold to 0.5 would have result in XGBoost correctly detecting the first SWR, and the 1D-CNN detecting the sharp-wave that has no ripple.

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
    - `merge_win`: Minimal length of the interval in miliseconds between predictions (`float`). If two detections are closer in time than this parameter, they will be merged together

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


### aux_fcn.process_LFP()

The python function `process_LFP(FP, sf, d_sf, channels)` of the `aux_fcn` module processes the LFP before it is input to the algorithm. It downsamples LFP to `d_sf`, and normalizes each channel separately by z-scoring them.

* Mandatory inputs:
	- `LFP`: LFP recorded data (`np.array`: `n_samples` x `n_channels`).
	- `sf`: sampling frequency (in Hz).
  	- `d_sf`: Desired subsampling frequency in Hz (`int`). By default all works in 1250 Hz, but can be changed if you retrain your models using `rippl_AI.retrain_model`.
	- `channels`: channel to which compute the undersampling and z-score normalization. Counting starts in `0`. If `channels` contains any `-1`, interpolation will be also applied. See `channels` of rippl_AI.predict(), or `aux_fcn.interpolate_channels()` for more information.

* Output:
	- `LFP_norm`: normalized LFP (`np.array`: `n_samples` x `len(channels)`). It is undersampled to 1250Hz, z-scored, and transformed to used the channels specified in `channels`.


### aux_fcn.interpolate_channels()

The python function `interpolate_channels(LFP, channels)` of the `aux_fcn` module allows creating more intermediate channels using interpolation. 

Because these models best performed using a richer spatial profile, all combinations of `architectures` and `model_numbers`  **work with 8 channels**. There is only one exception, for `architecture = 2D-CNN` with `models = {3, 4, 5}`, that needs to have **3 channels**. However, some times it's not possible to get such number of channels in the pyramidal layer, like when using linear probes (only 2 oe 3 channels fit in the pyramidal layer) or tetrodes (there are 4 recording channels). For this, we developed this interpolation function, that creates new channels between any pair of your recording channels. Using this approach, we can successfully use the already built algorithms with an equally high performance.

* Mandatory inputs:
	- `LFP`: LFP recorded data (`np.array`: `n_samples` x `n_channels`).
	- `channels`: list of channels over which to make interpolations (`np.array` or `list`: 1 x `# channels needed by the model` - 8 in most cases). Interpolated channels will be created in the positions of the `-1` elements of the list. Examples:
		- Let's say we have only 4 channels, so `LFP` is `n_samples` x 4. We can interpolate to get 8 functional channels. We will interpolate 1 channel between the first two, another one between 2nd and 3rd, and two more interpolated channels between the last two: 
			```
			# Define channels
			channels_interpolation = [0,-1,1,-1,2,-1,-1,3]

			# Make interpolation
			LFP_interpolated = aux_fcn.interpolate_channels(LFP, channels_interpolation)
			```
		- Let's say we have 8 channels, but channels 2 and 5 are dead. Then we want to interpolate them to get 8 fuctional channels:
			```
			# Define channels
			channels_interpolation = [0,1,-1,3,4,-1,6,7,8]

			# Make interpolation
			LFP_interpolated = aux_fcn.interpolate_channels(LFP, channels_interpolation)
			```
		- More usage examples can be found in the [examples_detection.ipynb](https://github.com/PridaLab/rippl-AI/blob/main/examples_detection.ipynb) python notebook.

* Output:
	- LFP_interpolated: Interpolated LFP (`np.array`: `n_samples` x `len(channels)`).


### aux_fcn.manual_curation()

The python function `aux_fcn.manual_curation(events, data, file_path, win_size=100, gt_events=None, sf=1250)` of the `aux_fcn` module allows doing a manual curation of the detected events. It displays an interactive GUI to manually select/discard the events. 

* Mandatory inputs:
	- `events`: array with events begining and end times in seconds (`2`,`n_det`).
	- `data`: normalized array with the input data (`n,n_channels`)
	- `file_path`: absolute path of the folder where the .txt with the curated predictions will be saved (`str`).
	- `win_size`: length of the displayed ripples in miliseconds (`int`)
	- `gt_events`: ground truth events beginning and end times in seconds (`2`,`n_gt_events`)
	- `sf`: sampling frequency (Hz) of the data/model output (`int`). Change if different than 1250 Hz.

 * Output: It always writes the curated events begin and end times in file_path.
	- curated_ids: boolean array with `True` for events that have been selected, and `False` for events that had been discarded (`#events`,)

Use cases:
1. If no GT events are provided, a the detected events will be provided, you can select which ones you want to keep (highligted in green) and which ones to discard (in red)
2. If GT events are provided, true positive detections (TP) will be displayed in green. If for any reason you want to discard correct detections, they will be displayed in yellow  


### aux_fcn.plot_all_events()

The python function `aux_fcn.plot_all_events(t_events, lfp, sf, win=0.1, title='', savefig='')` of the `aux_fcn` module plots all events in a single plot. It can be used as a fast summary/check after detection and/or curation.

* Mandatory inputs:
	- `events`: numpy array of size (`#events`, `1`) with all times of events
	- `lfp`: formated lfp with all channels
	- `sf`: sampling frequency of `lfp`

* Optional inputs:
    - `win`: window size at each side of the center of the ripple (`float`)
    - `title`: if provided, displays this title (`string`)
    - `savefig`: if provided, saves the image in the savefig directory (`string`).Full name required: e.g. images/session1_events.png


### aux_fcn.get_performance()

The python function `get_performance(predictions, true_events, threshold=0, exclude_matched_trues=False, verbose=True)` of the `aux_fcn` module computes several performance metrics:
* precision: also called *positive predictive value* is computed as (# good detections) / (# all detections)
* recall: also called *sensitivity* is computed as (# good detections) / (# all ground truth events)
* F1: computed as the harmonic mean between precision and recall, is a conservative and fair measure of performance. If any of precision or recall is low, F1 will be low. F1=1 only happens if detected events exactly match ground truth events.

Therefore, this function can be used only when some ground truth (i.e. events that we are considering the _truth_) is given. In order to check if a true event has been predicted, it computes the **Intersection over Union** (IoU). This index metric measures how much two intervals *intersect* with respect of the *union* of their size. So if `pred_events = [[2,3], [6,7]]` and `true_events = [[2,4]],[8,9]]`, then we would expect that the `IoU(pred_events[0], true_events[0]) > 0`, while the rest will be zero. 

* Mandatory inputs:
	- `predictions`: detected events (`np.array`: `n_predictions` x 2). First column are beginnings of the events (in seconds), second columns are ends of events (in seconds). This should be the output of `rippl_AI.get_intervals()`.
	- `true_events`: ground truth events (`np.array`: `n_groundtruth` x 2). Same format as `predictions`

* Optional inputs:
	- `threshold`: Threshold for the IoU (`bool`). By default is 0, so any intersection will be consider a match.
	- `exclude_matched_trues`: Boolean to determine if true events that had been already match to one prediction can be considered for other predicted events (`bool`). By default is `False`, so one true can match many predictions.
	- `verbose`: Print results (`bool`).

* Output:
	- `precision`: Metric indicating the percentage of correct predictions out of total predictions
	- `recall`: Metric indicating the percentage of true events predicted correctly
	- `F1`: Metric with a measure that combines precision and recall.
	- `TP`: True Positives (`np.array`: `n_predictions` x 1). It indicates which `pred_event` **detected** a `true_event`, so `True` are true positives, and `False` are false negatives.
	- `FN`: False Negatives (`np.array`: `n_groundtruth` x 1). It indicates which `true_event` was **not detected** by `pred_event`, so `True` are false negatives, and `False` are true positives.
	- `IOU`: IoU matrix (`np.array`: `n_predictions` x `n_groundtruth`). This can be used to know the matching indexes between `pred_event` and `true_event`.


## Re-training

Here, we provide a unique toolbox to easily re-train `model`s and adapt them to new datasets. These models have been selected because their architectural parameters are best fit to look for electrophysiological high-frequency events. So both if you are interested in finding SWRs or other electrophysiological events, these toolbox offers you the possility to skip all the parametric search and parameter tuning just by running this scripts. The advantages of the re-training module are:
* Avoid starting from scratch in making **your own feature-based detection algorithm**
* Easily plug-and-play to re-train already tested algorithms
* **Extend detection to other events** such as pathological fast ripples or interictal spikes
* **Extend detection to human** recordings


### rippl_AI.retrain_model()

The python function `rippl_AI.retrain_model(train_data, train_GT, test_data, test_GT, arch, parameters=None, save_path=None, d_sf=1250, merge_win=0)` of the `rippl_AI` module re-trains the best model of a given `architecture` to re-learn the optimal features to detect the new ground truth events annotated in the ground truth events.

* Mandatory inputs:
	- `train_data`: LFP recorded data that will be used to train the model (`np.array`: `n_samples` x `n_channels`). If several sessions needed, concatenate them to get the specified format.
	- `train_GT`: ground truth events corresponding to the `train_data` (`np.array`: `n_events` x 2). If several sessions were used, don't forget to readjust the times to properly refer to `train_data`.. Same format as `predictions`.
	- `test_data`: LFP recorded data that will be used to test the re-trained model (`list()` of `np.array`: `n_samples` x `n_channels`).
	- `test_GT`: ground truth events corresponding to the `test_data` (`list()` of `np.array`: `n_events` x 2). Event times refer to each element of the `test_data` list.

* Optional inputs:
	- `arch`: Name of the AI architecture to use (`string`). It can be: `CNN1D`, `CNN2D`, `LSTM`, `SVM` or `XGBOOST`.
	- `parameters`: dictionary, with the parameters that will be use in each specific architecture retraining
                - In 'XGBOOST': not needed
                - In 'SVM':     
                    parameters['Undersampler proportion']. Any value between 0 and 1. This parameter eliminates 
                                    samples where no ripple is present untill the desired proportion is achieved: 
                                    Undersampler proportion= Positive samples/Negative samples
                - In 'LSTM', 'CNN1D' and 'CNN2D': 
                    parameters['Epochs']. The number of times the training data set will be used to train the model
                    parameters['Training batch']. The number of windows that will be processed before updating the weights
	- `save_path`: string, path where the retrained model will be saved
	- `d_sf`: Desired subsampling frequency in Hz (`int`). By default all works in 1250 Hz, but this function allows using different subsampling frequencies.
	- `merge_win`: Minimal length of the interval in miliseconds between predictions (`float`). If two detections are closer in time than this parameter, they will be merged together

Usage examples can be found in the [examples_retraining.ipynb](https://github.com/PridaLab/rippl-AI/blob/main/examples_retraining.ipynb) python notebook.


## Exploration

Finally, as a further explotation of this toolbox, we also offer an exploration module, in which you can create your own model. In the [examples_explore](https://github.com/PridaLab/rippl-AI/blob/main/examples_explore/) folder, you can see how different architectures can be modified by multiple parameters to create infinite number of other models, that can be better adjusted to the need of your desired events. For example, if you are interested in lower frequency events, such as theta cycles, this exploratory module will be of utmost convenience to find an AI architecture that better adapts to the need of your research. Here, we specify the most common parameters to explore for each architecture:

### 1D-CNN
* Channels: number of LFP channel 
* Window size: LFP window size to evaluate: LFP window size to evaluate
* Kernel factor
* Batch size
* Number of epochs

### 2D-CNN
* Channels: number of LFP channel 
* Window size: LFP window size to evaluate

### LSTM
* Channels: number of LFP channel 
* Window size: LFP window size to evaluate
* Bidirectionality
* Number of layers
* Number of units per layer
* Number of epochs

### SVM
* Channels: number of LFP channel 
* Window size: LFP window size to evaluate
* Undersampling

### XGBoost
* Channels: number of LFP channel 
* Window size: LFP window size to evaluate
* Maximum tree depth
* Learning rate
* Gamma
* Lambda regularity
* Scale


# Enviroment setup

1. Install miniconda, following the tutorial: https://docs.conda.io/en/latest/miniconda.html
2. Launch the anaconda console, typing anaconda promp in the windows/linux search bar.
3. In the anaconda prompt, create a conda environment (e.g. `ripple_AI_env`):
```
conda create -n rippl_AI_env python=3.9.15
```
4. This will create a enviroment in your miniconda3 enviroments folder, usually: `C:\Users\<your_user>\miniconda3\envs`
5. Check that the enviroment `rippl_AI_env` has been created by typing:
```
conda env list
```
6. Activate the enviroment with:
```conda activate rippl_AI_env```
In case you want to launch the scripts from the command prompt. If you are using Visual Studio Code, you need to select the python interpreter `rippl_AI_env`
7. Next step after activating the enviroment, is to install every necessary python package:
```
conda install pip
pip install tensorflow==2.11 keras==2.11 xgboost==1.6.1 imblearn numpy matplotlib pandas scipy
pip install -U scikit-learn==1.1.2
```
To download the lab data from figshare (not normalized, sampled with the original frequency of 30 000 Hz):
```
git clone https://github.com/cognoma/figshare.git
cd figshare
python setup.py install
```
The package versions compatible with the toolbox are:

- h5py==3.11.0
- imbalanced-learn==0.12.2
- imblearn==0.0
- ipython==8.18.1
- keras==2.11.0
- numpy==1.26.4
- pandas==2.2.2
- pip==23.3.1
- python==3.9.19
- scikit-learn==1.1.2
- scipy==1.13.0
- tensorflow==2.11.0
- xgboost==1.6.1
