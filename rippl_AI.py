import numpy as np
import matplotlib.style as mplstyle
mplstyle.use('fast')
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import keras
import os
from aux_fcn import process_LFP,prediction_parser, get_predictions_index, middle_stamps,get_click_th, format_predictions,split_data, retraining_parser, save_model,get_performance

# Detection functions

def predict(LFP,sf,d_sf=1250,arch='CNN1D',model_number=1,channels=np.arange(8),new_model=None,n_channels=None,n_timesteps=None):
    ''' 
    predict(LFP,sf,d_sf=1250,arch='CNN1D',model_number=1,channels=np.arange(8),new_model=None,n_channels=None,n_timesteps=None)

    Returns the requested architecture and model number output probability

    Mandatory inputs:
    -----------------
        LFP: (np.array: n_samples x n_channels). LFP_recorded data. Although there 
              are no restrictions in n_channels, some considerations should be taken into 
              account (see channels). Data does not need to be normalized, because it will 
              be internally be z-scored (see aux_fcn.process_LFP())
        sf: (int) Original sampling frequency (in Hz)

    Optional inputs:
    ----------------
        d_sf: (int) Desired subsampling frequency (in Hz)
        arch: Name of the AI architecture to use (string). 
              It can be: CNN1D, CNN2D, LSTM, SVM or XGBOOST.
        model_number: Number of the model to use (integer). There are six different models 
              for each architecture, sorted by performance, 1 being the best, and 5 the last. 
              A sixth model is included if single-channel data needs to be used.
        channels: Channels to be used for detection (np.array or list: 1 x 8). This is the most 
              senstive parameter, because models will be looking for specific spatial features 
              over all channels. Counting starts in 0. 
              The two main remarks are: 
                - All models have been trained to look at features in the pyramidal layer (SP), 
                  so for them to work at their maximum potential, the selected channels would  
                  ideally be centered in the SP, with a postive deflection on the first channels  
                  (upper channels) and a negative deflection on the last channels (lower channels).  
                - For all combinations of architectures and model_numbers, channels has to be  
                  of size 8. There is only one exception, for architecture = 2D-CNN with  
                  models = {3, 4, 5}, that needs to have 3 channels.
                - If you are using a high-density probe, then we recommend to use equi-distant 
                  channels from the beginningto the end of the SP. For example, for Neuropixels 
                  in mice, a good set of channels would be pyr_channel + [-8,-6,-4,-2,0,2,4,6].
                - In the case of linear probes or tetrodes, there are not enough density to cover 
                  the SP with 8 channels. For that, interpolation or recorded channels can be 
                  done without compromising performance. New artificial interpolated channels 
                  will be add to the LFP wherever there is a -1 in channels.
                  For example, if pyr_channel=11 in your linear probe, so that 10 is in stratum 
                  oriens and 12 in stratum radiatum, then we could define channels=[10,-1,-1,11,-1,-1,-1,12],
                  where 2nd and 3rd channelswill be an interpolation of SO and SP channels, and 
                  5th to 7th an interpolation of SP and SR channels.For tetrodes, organising 
                  channels according to their spatial profile is very convenient to assure best 
                  performance. These interpolations are done using the function aux_fcn.interpolate_channels().
        new_model: Other re-trained model you want to use for detection. If you have used our re-train function 
              to adapt the optimized models to your own data (see rippl_AI.retrain() for more details),
              you can input the new_model here to use it to predict your events.
              IMPORTANT! If you are using new_model, the data wont be processed, so make sure to 
                         have your data z-scored, subsampled at your subsampling freq and with the 
                         correct channels before calling predict, for example using the process_LFPfunction  
              IMPORTANT! If you are using a new_model, you have to pass as arguments its number of 
                         channels and the timesteps for window
        n_channels: (int) the number of channels of the new model
        timesteps: (int) the number of timesteps per window of the new model

    Output:
    -------
        SWR_prob: model output for every sample of the LFP (np.array: n_samples x 1). 
                 It can be interpreted as the confidence or probability of a SWR event, so values 
                 close to 0 mean that the model is certain that there are not SWRs, and values close 
                 to 1 that the model is very sure that there is a SWR hapenning.
        LFP_norm: LFP data used as an input to the model (np.array: n_samples x len(channels)). 
                 It is undersampled, z-scored, and transformed to used the channels specified in channels.

    A Rubio, 2023 LCN
    '''

    #channels=opt['channels']
    if new_model==None:
        norm_LFP=process_LFP(LFP,sf,d_sf,channels)
    else: # Data is supossedly already normalized when using new model
        norm_LFP=LFP
    prob=prediction_parser(norm_LFP,arch,model_number,new_model,n_channels,n_timesteps)

    return(prob,norm_LFP)


def predict_ens(ens_input,model_name='ENS'):
    '''
    predict_ens(ens_input,model_name='ENS')

    Generates the output of the ensemble model specified with the model name
    
    Inputs: 
    -------
        ens_input: (n_samples, 5) input of the model, consisting of the inputs of 
            the 5 other different architectures
        model_name:  str, name of the ens model found in the folder 'optimized_models'

    Outputs:
    --------
        prob: (n_samples) output of the model, the calculated probability of 
            an event in each sample
    
    '''
    model = keras.models.load_model(os.path.join('optimized_models',model_name))
    prob = model.predict(ens_input,verbose=1)
    return prob



def get_intervals(y,threshold=None,LFP_norm=None,sf=1250,win_size=100,file_path=None,merge_win=0):
    ''' 
    get_intervals(y,LFP_norm=None,sf=1250,win_size=100,threshold=None,file_path=None)
    
    Get events initial and end times, in seconds
    Displays a GUI to help you select the best threshold.

    Inputs:
    -------
        y: (n,) one dimensional output signal of the model
        threshold: (float), threshold of predictions
        LFP_norm: (n,n_channels), normalized input signal of the model
        sf: (int), sampling frequency (Hz) of the LFP_norm/model output. 
            Change if used is different than 1250
        win_size: (int), length of the displayed ripples in miliseconds
        file_path: (str), absolute path of the folder where the .txt with the predictions 
            will be generated. Leave empty if you don't want to generate the file
        merge_win: (float), minimal length of the interval in miliseconds between predictions. If 
            two detections are closer in time than this parameter, they will be merged together
    

    Output:
    -------

        predictions: (n_events,2), returns the time (seconds) of the begining and end of each event
            4 possible use cases, depending on which parameter combination is used when calling the function.
            1.- (y): a histogram of the output is displayed, you drag a vertical bar to select your th
            2.- (y,th): no GUI is displayed, the predictions are gererated automatically
            3.- (y,LFP_norm): some examples of detected events are displayed next to the histogram
            4.- (y,LFP_norm,th): same case as 3, but the initial location of the bar is th
    
    '''
    global predictions_index, line
    # Merge samples
    merge_s=round(sf*merge_win/1000)
    # If LFP_norm is passed, plot detected ripples
    if type(LFP_norm)==np.ndarray:

        timesteps=int((win_size*sf/1000)//2)
        if threshold==None:   
            valinit=0.5
        else:
            valinit=threshold
        # The predictions_index with the initial  th=0.5 is generated

        fig, axes = plt.subplot_mosaic("AAAAABCDEF;AAAAAGHIJK;AAAAALMNÑO;AAAAAPQRST;AAAAAUVWXY",figsize=(15,6))
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.suptitle(f"Threshold selection")
        axes['A'].set_title(f'Th: {valinit}')
        for key in axes.keys():
            if key=='A':
                axes['A'].hist(y)
                axes['A'].set_yscale('log')
                line=axes['A'].axvline(valinit,c='k')
                continue
            axes[key].set_yticklabels([])
            axes[key].set_xticklabels([])
            axes[key].set_xticks([])
            axes[key].set_yticks([])
        axcolor = (20/255,175/255,245/255) # light blue
        hovercolor=(214/255,255/255,255/255)
        
        # Plot button definition
        plotax = plt.axes([0.4, 0.53, 0.035, 0.04])
        button_plot = Button(plotax, 'Plot', color=axcolor, hovercolor=hovercolor)
        # Save button definition
        Saveax = plt.axes([0.375, 0.47, 0.095, 0.04])

        button_save = Button(Saveax, f'Save: {len(get_predictions_index(y,valinit,merge_samples=merge_s))} events', color=axcolor, hovercolor=hovercolor)

        def plot_ripples():
            global predictions_index

            th=line.get_xdata()[0]
            
            predictions_index=get_predictions_index(y,th,merge_samples=merge_s)
            n_pred=len(predictions_index)
            # Clearing the axes
            for key in axes.keys():
                if key=='A':
                    continue
                else:
                    axes[key].clear()
                    axes[key].set_yticklabels([])
                    axes[key].set_xticklabels([])
                    axes[key].set_xticks([])
                    axes[key].set_yticks([])
                    
            
            if n_pred==0:
                print("No predictions with this threshold")
                return
            else:
                mids=middle_stamps(predictions_index)
                pos_mat = list(range(LFP_norm.shape[1]-1, -1, -1)) * np.ones((timesteps*2, LFP_norm.shape[1]))
                if len(mids)<25:
                    for i,key in enumerate(axes.keys()):
                        if key=='A':
                            continue
                        if i>len(mids): # End of the events 
                            break
                        # De momento quito la normalización, se va a la mierda muy a menudo
                        extracted_window=LFP_norm[mids[i-1]-timesteps:mids[i-1]+timesteps,:]
                        x=extracted_window*1/np.max(extracted_window)+pos_mat

                        lines=axes[key].plot(x,c='0.6',linewidth=0.5)
                        # Ripple fragment different color
                        ini_rip=int(predictions_index[i-1,0]) # Timestamps absolutos
                        end_rip=int(predictions_index[i-1,1])
                        small_pos_mat = list(range(LFP_norm.shape[1]-1, -1, -1)) * np.ones((end_rip- ini_rip, LFP_norm.shape[1]))
                        ripple_window=LFP_norm[ini_rip:end_rip,:]
                        
                        x_ripple=ripple_window*1/np.max(extracted_window)+small_pos_mat
                        samples_ripple=np.linspace(ini_rip,end_rip,end_rip-ini_rip,dtype=int)-(mids[i-1]-timesteps)

                        rip_lines=axes[key].plot(samples_ripple,x_ripple,c='k',linewidth=0.5)

                else: # More than 25 events: Random selection of 25 events
                    sample_index=np.random.permutation(len(mids))[:25]
                    for i,key in enumerate(axes.keys()):
                        if key=='A':
                            continue

                        extracted_window=LFP_norm[mids[sample_index[i-1]]-timesteps:mids[sample_index[i-1]]+timesteps,:]

                        x=extracted_window*1/np.max(extracted_window)+pos_mat

                        lines=axes[key].plot(x,c='0.6',linewidth=0.5)
                        # Ripple fragment different color
                        ini_rip=int(predictions_index[sample_index[i-1],0]) # Timestamps absolutos
                        end_rip=int(predictions_index[sample_index[i-1],1])
                        small_pos_mat = list(range(LFP_norm.shape[1]-1, -1, -1)) * np.ones((end_rip- ini_rip, LFP_norm.shape[1]))
                        ripple_window=LFP_norm[ini_rip:end_rip,:]
                        
                        x_ripple=ripple_window*1/np.max(extracted_window)+small_pos_mat
                        samples_ripple=np.linspace(ini_rip,end_rip,end_rip-ini_rip,dtype=int)-(mids[sample_index[i-1]]-timesteps)

                        rip_lines=axes[key].plot(samples_ripple,x_ripple,c='k',linewidth=0.5)
            plt.draw()

        plot_ripples()                
        # button generate events ripple 
        
        def generate_pred(event):
            global predictions_index
            # Generar las predicciones con el th guardado
            th=line.get_xdata()[0]
            predictions_index=get_predictions_index(y,th,merge_samples=merge_s)

            if file_path:  
                format_predictions(file_path,predictions_index,sf)
            plt.close()
            return
        button_save.on_clicked(generate_pred)
        # Plot random ripples
        ############################
        # Click events
        def on_click_press(event):
            global line
            if event.button is MouseButton.LEFT:
                clicked_ax=event.inaxes
                if clicked_ax==axes['A']:
                    th=get_click_th(event)
                    line.remove()
                    line=axes['A'].axvline(th,c='k')
                    clicked_ax.set_title(f'Th: {th:1.3f}')
                    n_events=len(get_predictions_index(y,th,merge_samples=merge_s))
                    button_save.label.set_text(f"Save: {n_events} events")

        plt.connect('button_press_event',on_click_press)
        plt.connect('motion_notify_event',on_click_press)
        def on_click_release(event):
            if event.button is MouseButton.LEFT:
                clicked_ax=event.inaxes
                if clicked_ax==axes['A']:
                    plot_ripples()
        plt.connect('button_release_event',on_click_release)

        def plot_button_click(event):
            # Generar las predicciones otra vez
            plot_ripples()

        button_plot.on_clicked(plot_button_click)
        plt.show(block=True)

    # If no threhold is defined, choose your own with the GUI,without LFP_norm plotting
    elif threshold==None:
        axcolor = (20/255,175/255,245/255) # light blue
        hovercolor=(214/255,255/255,255/255)
        valinit=0.5
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(y)
        ax.set_yscale('log')
        fig.suptitle(f"Threshold selection")
        ax.set_title(f'Th: {valinit}')

        line=ax.axvline(valinit,c='k')
        # Button definition
        resetax = plt.axes([0.7, 0.5, 0.12, 0.075])
        button = Button(resetax, f'Save\n{len(get_predictions_index(y,valinit,merge_samples=merge_s))} events', color=axcolor, hovercolor=hovercolor)


        # Button definition
        
        def generate_pred(event):
            global predictions_index
            th=line.get_xdata()[0]
            predictions_index=get_predictions_index(y,th,merge_samples=merge_s)
            if file_path:  # Si la linea del archivo no esta vacia
                format_predictions(file_path,predictions_index,sf)
            plt.close()
            
        button.on_clicked(generate_pred)
        
        
        def on_click(event):
            global line
            if event.button is MouseButton.LEFT:
                clicked_ax=event.inaxes
                if clicked_ax==ax:
                    th=get_click_th(event)
                    line.remove()
                    line=ax.axvline(th,c='k')
                    ax.set_title(f'Th: {th:1.3f}')

                    n_events=len(get_predictions_index(y,th,merge_samples=merge_s))
                    button.label.set_text(f"Save\n{n_events} events")
                    plt.draw()
        plt.connect('button_press_event',on_click)

        plt.connect('motion_notify_event', on_click)
        plt.show(block=True)
    # If threshold is defined, and no LFP_norm is passsed, the function simply generates the predictions     
    else:
        predictions_index=get_predictions_index(y,threshold,merge_samples=merge_s)
        if file_path:
            format_predictions(file_path,predictions_index,sf)
    return (predictions_index/sf)

# Prepares data for training, used in retraining and exploring notebooks
def prepare_training_data(train_LFPs,train_GTs,val_LFPs,val_GTs,sf=30000,d_sf=1250,channels=np.arange(0,8)):
    '''
    prepare_training_data(train_LFPs,train_GTs,val_LFPs,val_GTs,sf=30000,d_sf=1250,channels=np.arange(0,8))

    Prepares data for training: subsamples, interpolates (if required), z-scores and concatenates 
    the train/test data passed. Does the same for the validation data, but without concatenating

    Inputs:
    -------
        train_LFPs:  (n_train_sessions) list with the raw LFP of n sessions that will be used to train
        train_GTs:   (n_train_sessions) list with the GT events of n sessions, in the format [ini end] in seconds
        (A): quizá se podría quitar esto, lo de formatear tambien las de validacion 
        val_LFPs:    (n_val_sessions) list: with the raw LFP of the sessions that will be used in validation
        val_GTs:     (n_val_sessions) list: with the GT events of n validation sessions
        sf:          (int) original sampling frequency of the data in Hz
        sf:          (int) desired downsample frequency of the data in Hz
        channels:    (n_channels) np.array. Channels that will be used to generate data. Check interpolate_channels for more information
    
    Output:
    -------
        retrain_LFP: (n_samples x n_channels): sumbsampled, z-scored, interpolated and concatenated data from all the training sessions
        retrain_GT:  (n_events x 2): concatenation of all the events in the training sessions
        norm_val_GT: (n_val_sessions) list: list with the normalized LFP of all the val sessions
        val_GTs:     (n_val_sessions) list: Gt events of each val sessions

    A Rubio LCN 2023

    '''
    assert len(train_LFPs) == len(train_GTs), "The number of train LFPs doesn't match the number of train GTs"
    assert len(val_LFPs) == len(val_GTs), "The number of test LFPs doesn't match the number of test GTs"

    # All the training sessions data and GT will be concatenated in one data array and one GT array (2 x n events)
    retrain_LFP=[]
    for LFP,GT in zip(train_LFPs,train_GTs):
        # 1st session in the array
        print('Original training data shape: ',LFP.shape)
        if retrain_LFP==[]:
            retrain_LFP=process_LFP(LFP,sf,d_sf,channels)
            offset=len(retrain_LFP)/d_sf
            retrain_GT=GT
        # Append the rest of the sessions, taking into account the length (in seconds) 
        # of the previous sessions, to cocatenate the events' times
        else:
            aux_LFP=process_LFP(LFP,sf,d_sf,channels)
            retrain_LFP=np.vstack([retrain_LFP,aux_LFP])
            retrain_GT=np.vstack([retrain_GT,GT+offset])
            offset+=len(aux_LFP)/d_sf
    # Each validation session LFP will be normalized, etc and stored in an array
    #  the GT needs no further treatment
    norm_val_GT=[]
    for LFP in val_LFPs:
        print('Original validation data shape: ',LFP.shape)
        norm_val_GT.append(process_LFP(LFP,sf,d_sf,channels))


    return retrain_LFP, retrain_GT , norm_val_GT, val_GTs

# Retrain the best model of each architecture, and save it in the path specified in save_path.
#  also plots the trai, test and validation performance
def retrain_model(LFP_retrain,GT_retrain,LFP_val,GT_val,arch,parameters=None,save_path=None,d_sf=1250,merge_win=0):
    '''
    retrain_model(LFP_retrain,GT_retrain,LFP_val,GT_val,arch,parameters=None,save_path=None,d_sf=1250,merge_win=0)

    Retrains the best model of the specified architecture with the retrain data and the specified parameters. Performs validation if validation data is provided, and plots the train, test and validation performance.
    
    Mandatory inputs:
    -----------------
        LFP_retrain: (n_samples x n_channels). Concatenated LFP of all the trained sessions
        GT_retrain:  (n_events x 2). List with the concatenated GT events times of n sessions, 
                     in the format [ini end] in seconds
        LFP_val:     (n_val_sessions). List with the normalized LFP of the sessions that will
                     be used in validation
        GT_val:      (n_val_sessions). List with the GT events of the validation sessions
        arch:        (string). Architecture of the model to be retrained

    Optional inputs:
    ----------------
        parameters:  (dictionary) Parameters that will be use in each specific architecture retraining
            - For 'XGBOOST', not needed.
            - For 'SVM', one parameter is needed:
                    - parameters['Undersampler proportion']: Any value between 0 and 1. 
                        This parameter eliminates samples where no ripple is present untill the 
                        desired proportion is achieved: 
                        Undersampler proportion= Positive samples/Negative samples
            - For 'LSTM', 'CNN1D' and 'CNN2D', two things are needed:
                    - parameters['Epochs']. The number of times the training data set will
                        be used to train the model.
                    - parameters['Training batch']. The number of windows that will be processed 
                        before updating the weights
        save_path: (string). Path where the retrained model will be saved
        d_sf:      (int) Desired subsampling frequency (in Hz)
        merge_win: (float). Minimal length of the interval in miliseconds between predictions. If 
                    two detections are closer in time than this parameter, they will be merged together

    
    Output:
    -------
        retrain_LFP: (n_samples x n_channels): sumbsampled, z-scored, interpolated and concatenated data from all the training sessions
        retrain_GT:  (n_events x 2): concatenation of all the events in the training sessions
        norm_val_GT: (n_val_sessions) list: list with the normalized LFP of all the val sessions
        val_GTs:     (n_val_sessions) list: Gt events of each val sessions
    
    A Rubio LCN 2023

    '''
    merge_s=round(d_sf*merge_win/1000)
    # Do the train/test split. Feel free to try other proportions
    LFP_test,events_test,LFP_train,events_train=split_data(LFP_retrain,GT_retrain,split=0.7,sf=d_sf)

    print(f'Number of validation sessions: {len(LFP_val)}') #TODO: for shwoing length and events
    print(f'Shape of train data: {LFP_train.shape}, Number of train events: {events_train.shape[0]}')
    print(f'Shape of test data: {LFP_test.shape}, Number of test events: {events_test.shape[0]}')

    # prediction parser returns the retrained model, the output predictions probabilities
    model,y_pred_train,y_pred_test=retraining_parser(arch,LFP_train,events_train,LFP_test,events_test,params=parameters,d_sf=d_sf)

    # Save model if save_path is not empty
    if save_path:
        save_model(model,arch,save_path)

    # Plot section #
    # for loop iterating over the validation data
    val_pred=[]
    # The correct n_channels and timesteps needs to be passed to predict for the fcn to work when using new_model
    if arch=='XGBOOST':
        n_channels=8
        timesteps=16
    elif arch=='SVM':
        n_channels=8
        timesteps=1
    elif arch=='LSTM':
        n_channels=8
        timesteps=32
    elif arch=='CNN2D':
        n_channels=8
        timesteps=40
    elif arch=='CNN1D':
        n_channels=8
        timesteps=16
    
    for LFP in LFP_val:
        val_pred.append(predict(LFP,sf=d_sf,arch=arch,new_model=model,n_channels=n_channels,n_timesteps=timesteps)[0])
    # Extract and plot the train and test performance
    th_arr=np.linspace(0.1,0.9,9)
    F1_train=np.empty(shape=len(th_arr))
    F1_test=np.empty(shape=len(th_arr))
    for i,th in enumerate(th_arr):
        pred_train_events=get_predictions_index(y_pred_train,th,merge_samples=merge_s)/d_sf
        pred_test_events=get_predictions_index(y_pred_test,th,merge_samples=merge_s)/d_sf
        _,_,F1_train[i],_,_,_=get_performance(pred_train_events,events_train,verbose=False)
        _,_,F1_test[i],_,_,_=get_performance(pred_test_events,events_test,verbose=False)
    

    fig,axs=plt.subplots(1,2,figsize=(12,5),sharey='all')
    axs[0].plot(th_arr,F1_train,'k.-')
    axs[0].plot(th_arr,F1_test,'b.-')
    axs[0].legend(['Train','Test'])
    axs[0].set_ylim([0 ,max(max(F1_train), max(F1_test)) + 0.1])
    axs[0].set_title('F1 test and train')
    axs[0].set_ylabel('F1')
    axs[0].set_xlabel('Threshold')


    # Validation plot in the second ax
    F1_val=np.zeros(shape=(len(LFP_val),len(th_arr)))
    for j,pred in enumerate(val_pred):
        for i,th in enumerate(th_arr):
            pred_val_events=get_predictions_index(pred,th,merge_samples=merge_s)/d_sf
            _,_,F1_val[j,i],_,_,_=get_performance(pred_val_events,GT_val[j],verbose=False)

    for i in range(len(LFP_val)):
        axs[1].plot(th_arr,F1_val[i])
    axs[1].plot(th_arr,np.mean(F1_val,axis=0),'k.-')
    axs[1].set_title('Validation F1')
    axs[1].set_ylabel('F1')
    axs[1].set_xlabel('Threshold')
   

    
    plt.show()
