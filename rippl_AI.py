import numpy as np
import matplotlib.style as mplstyle
mplstyle.use('fast')
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from aux_fcn import process_LFP,prediction_parser, get_predictions_index, middle_stamps,get_click_th, format_predictions

# Detection functions

def predict(LFP,sf,arch='CNN1D',model_number=1,channels=np.arange(8)):
    ''' returns the requested architecture and model number output probability

    Mandatory inputs:
        LFP: (np.array: n_samples x n_channels). LFP_recorded data. Although there are no restrictions in n_channels, 
            some considerations should be taken into account (see channels)
            Data does not need to be normalized, because it will be internally be z-scored (see aux_fcn.process_LFP())
        sf: (int) sampling frequency (in Hz)
    Optional inputs:
        arch: Name of the AI architecture to use (string). It can be: CNN1D, CNN2D, LSTM, SVM or XGBOOST.
        model_number: Number of the model to use (integer). There are five different models for each architecture, 
            sorted by performance, being 1 the best, and 5 the last.
        channels: Channels to be used for detection (np.array or list: 1 x 8). This is the most senstive parameter,
            because models will be looking for specific spatial features over all channels. Counting starts in 0. 
            The two main remarks are: 
                - All models have been trained to look at features in the pyramidal layer (SP), so for them to work 
                    at their maximum potential, the selected channels would ideally be centered in the SP, 
                    with a postive deflection on the first channels (upper channels) and a negative deflection on 
                    the last channels (lower channels). The image TODO (link a la imagen)
                - For all combinations of architectures and model_numbers, channels has to be of size 8. 
                    There is only one exception, for architecture = 2D-CNN with models = {3, 4, 5}, that needs to have 3 channels.
                -If you are using a high-density probe, then we recommend to use equi-distant channels from the beginning
                    to the end of the SP. For example, for Neuropixels in mice, a good set of channels would be 
                    pyr_channel + [-8,-6,-4,-2,0,2,4,6].
                - In the case of linear probes or tetrodes, there are not enough density to cover the SP with 8 channels. 
                    For that, interpolation or recorded channels can be done without compromising performance. 
                    New artificial interpolated channels will be add to the LFP wherever there is a -1 in channels.
                    TODO: igual el siguiente párrafo se pude quitar,
                    For example, if pyr_channel=11 in your linear probe, so that 10 is in stratum oriens and 12 in 
                    stratum radiatum, then we could define channels=[10,-1,-1,11,-1,-1,-1,12], where 2nd and 3rd channels
                    will be an interpolation of SO and SP channels, and 5th to 7th an interpolation of SP and SR channels.
                    For tetrodes, organising channels according to their spatial profile is very convenient to assure best 
                    performance. These interpolations are done using the function aux_fcn.interpolate_channels().
        new_model: Other re-trained model you want to use for detection. If you have used our re-train function 
            to adapt the optimized models to your own data (see rippl_AI.retrain() for more details), you can input the new_model 
            here to use it to predict your events.
    Output:
        SWR_prob: model output for every sample of the LFP (np.array: n_samples x 1). It can be interpreted as the confidence 
            or probability of a SWR event, so values close to 0 mean that the model is certain that there are not SWRs,
            and values close to 1 that the model is very sure that there is a SWR hapenning.
        LFP_norm: LFP data used as an input to the model (np.array: n_samples x len(channels)). It is undersampled to 1250Hz, 
            z-scored, and transformed to used the channels specified in channels.
    A Rubio, 2023 LCN
    '''
    #channels=opt['channels']
    print(channels)
    print('Original LFP shape: ',LFP.shape)
    norm_LFP=process_LFP(LFP,sf,channels)
    prob=prediction_parser(norm_LFP,arch,model_number)

    return(prob,norm_LFP)

# Get events initial and end times, in seconds

def get_intervals(y,LFP_norm=None,sf=1250,win_size=100,threshold=None,file_path=None):

    ''' Displays a GUI to help you select the best threshold.
        Inputs: y :          (n,) one dimensional output signal of the model
                threshold:   float, threshold of predictions
                LFP_norm:        (n,n_channels), normalized input signal of the model
                file_path:   str, absolute path of the folder where the .txt with the predictions will be generated
                             Leave empty if you don't want to generate the file
                win_size:    int, length of the displayed ripples in miliseconds
                sf:          int, sampling frequency (Hz) of the LFP_norm/model output. Change if different than 1250
        Output: predictions: (n_events,2), returns the time (seconds) of the begining and end of each vents
        4 possible use cases, depending on which parameter combination is used when calling the function.
            1.- (y): a histogram of the output is displayed, you drag a vertical bar to selecct your th
            2.- (y,th): no GUI is displayed, the predictions are gererated automatically
            3.- (y,LFP_norm): some examples of detected events are displayed next to the histogram
            4.- (y,LFP_norm,th): same case as 3, but the initial location of the bar is th
    
    '''
    global predictions_index
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

        button_save = Button(Saveax, f'Save: {len(get_predictions_index(y,valinit))} events', color=axcolor, hovercolor=hovercolor)
        
        line.set_xdata(valinit) # Sin esta línea get_xdata devolvía un vector, lo que acía que la primera llamada a plot ripples fallara  Yokse

        def plot_ripples():
            th=line.get_xdata()
            
            predictions_index=get_predictions_index(y,th)
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
            th=line.get_xdata()
            predictions_index=get_predictions_index(y,th)

            if file_path:  
                format_predictions(file_path,predictions_index,sf)
            plt.close()
            return
        button_save.on_clicked(generate_pred)
        # Plot random ripples
        ############################
        # Click events
        def on_click_press(event):
            if event.button is MouseButton.LEFT:
                clicked_ax=event.inaxes
                if clicked_ax==axes['A']:
                    th=get_click_th(event)
                    line.set_xdata(th)
                    clicked_ax.set_title(f'Th: {th:1.3f}')
                    n_events=len(get_predictions_index(y,th))
                    button_save.label.set_text(f"Save: {n_events} events")
                    #plt.show()

        plt.connect('button_press_event',on_click_press)
        plt.connect('motion_notify_event',on_click_press)
        def on_click_release(event):
            if event.button is MouseButton.LEFT:
                clicked_ax=event.inaxes
                if clicked_ax==axes['A']:
                    plot_ripples()
                    #plt.show()
        plt.connect('button_release_event',on_click_release)

        def plot_button_click(event):
            # Generar las predicciones otra vez
            plot_ripples()

        button_plot.on_clicked(plot_button_click)
        plt.show()

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
        line.set_xdata(valinit)
        # Button definition
        resetax = plt.axes([0.7, 0.5, 0.12, 0.075])
        button = Button(resetax, f'Save\n{len(get_predictions_index(y,valinit))} events', color=axcolor, hovercolor=hovercolor)


        # Button definition
        
        def generate_pred(event):
            # Generar las predicciones con el th guardado
            global predictions_index
            th=line.get_xdata()
            predictions_index=get_predictions_index(y,th)
            if file_path:  # Si la linea del archivo no esta vacia
                format_predictions(file_path,predictions_index,sf)
            plt.close()
            
        button.on_clicked(generate_pred)
        
        
        def on_click(event):
            if event.button is MouseButton.LEFT:
                clicked_ax=event.inaxes
                if clicked_ax==ax:
                    th=get_click_th(event)
                    line.set_xdata(th)
                    ax.set_title(f'Th: {th:1.3f}')

                    n_events=len(get_predictions_index(y,th))
                    button.label.set_text(f"Save\n{n_events} events")
                    plt.draw()
        plt.connect('button_press_event',on_click)

        plt.connect('motion_notify_event', on_click)
        plt.show()
    # If threhold is defined, and no LFP_norm is passsed, the function simply generates the predictions     
    else:
        print(y,threshold)
        predictions_index=get_predictions_index(y,threshold)
        if file_path:
            format_predictions(file_path,predictions_index,sf)
    return (predictions_index/sf)
