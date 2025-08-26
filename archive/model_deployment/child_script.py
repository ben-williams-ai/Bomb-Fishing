# Import the required libraries from the master script
import tensorflow as tf
from tensorflow.keras.models import load_model
import autokeras
import librosa
import soundfile as sf
import os
import datetime
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import sys



def main():
    print("\n     Starting a new batch!\n")

    # Get the filename from the command-line arguments
    current_batch = sys.argv[1]
    new_audio_dir = sys.argv[2]
    output_folder = sys.argv[3]
    model_dir = sys.argv[4]
    sample_rate = int(sys.argv[5])
    batch_counter = (int(sys.argv[6]))+1
    total_num_batches = int(sys.argv[7])
    file_counter = int(sys.argv[8]) ### unsure how to use yet

    # load the tensorflow model
    best_model = load_model(model_dir)

    # recording progress
    skipped_files = []  ############################# integreate this!
    suspected_bombs = []
    checked_file_counter = 0
    #file_counter = 0


    #set the sample rate that should be used (this will trigger resampling of audio if needed)
    new_sample_rate = 8000

    # set the length of each window in samples
    window_length_samples = int(2.88 * new_sample_rate)
    
    # set the stagger to add to stream2 below in second for loop
    stagger = int(window_length_samples/2)

   # read current batch from file
    with open(current_batch, 'r') as f:
        current_batch_files = [line.strip() for line in f]

    # the main loop
    for f in tqdm(current_batch_files):
        try:
            # print progress
            checked_file_counter += 1
            print('\nBatch {} of {}. Checking file: {}'.format(str(batch_counter), str(total_num_batches), f))

            #load the audio
            audio_path = os.path.join(new_audio_dir, f)
            audio, sample_rate = librosa.load(path=audio_path, sr=sample_rate)

            # resample the audio if needed
            if sample_rate != new_sample_rate:
                audio = librosa.resample(audio, sample_rate, new_sample_rate)

            # Compute the number of windows
            num_windows = len(audio) // window_length_samples

            # for skipping the first stream if needed later in next for loop
            timestamp2 = None

            # count windows to record how many were checked if the last was skipper
            window_counter = 0

            # Iterate over windows and compute the MFCC spectrogram for each window
            for i in range(num_windows):

                """This is done in 2 streams. Stream1 takes starts at 0sec and 
                    takes a window every 2.88 seconds. Stream2 starts at 1.44sec and
                    take a window every 2.88 seconds from here. As bombs are 2.88 seconds
                    in length maximum, this ensures no bomb that would be cut in half by
                    stream1 is missed, as stream 2 has a staggered window."""

                # Get the start and end indices for the current window
                start_index1 = i * window_length_samples
                start_index2 = (i * window_length_samples) + stagger

                end_index1 = (i + 1) * window_length_samples
                end_index2 = ((i + 1) * window_length_samples) + stagger

                # Extract the current window
                window1 = audio[start_index1:end_index1]
                window2 = audio[start_index2:end_index2]

                # check window length fits network input shape
                window_counter += 1

                # Compute the MFCC spectrogram for the current window
                mfcc_spec1 = librosa.feature.mfcc(y=window1, sr=new_sample_rate, n_mfcc = 32)
                mfcc_spec2 = librosa.feature.mfcc(y=window2, sr=new_sample_rate, n_mfcc = 32)

                # add extra dimensions for networks input_shape
                mfcc_spec1 = np.expand_dims(mfcc_spec1, axis=2)
                mfcc_spec2 = np.expand_dims(mfcc_spec2, axis=2)
                mfcc_spec1 = np.expand_dims(mfcc_spec1, axis=0)
                mfcc_spec2 = np.expand_dims(mfcc_spec2, axis=0)

                # TODO(hamer): investigate batch inference.
                # Perform the function on the MFCC spectrogram
                result1 = best_model.predict(mfcc_spec1, verbose=0) > 0.5
                if len(window2) == 23040:
                    result2 = best_model.predict(mfcc_spec2, verbose=0) > 0.5
                else:
                    result2 = None
                    #print('\n    Checked {} windows. Skipping last window as its <1.44sec'.format(window_counter)) ######################### not printing where it should

                # check if a bomb was found in the prev 1.44 second window for stream 2
                check = round((start_index1/new_sample_rate) - 1.44, 2)

                # if stream 1 is a bomb, and it was not found by the preceding stream 2 (skip this step if its the very first window)
                if result1 == True and timestamp2 != check:# and timestamp2 is not None:
                    # get timestamp of this window 
                    timestamp1 = start_index1/new_sample_rate
                    timestamp_str1 = str(datetime.timedelta(seconds=timestamp1))
                    hmmss = timestamp_str1[0:7] # remove milliseconds
                    print('Suspected bomb at: ' + hmmss)

                    # select the 5 second window to save
                    # max is used to ensure the window isnt set to a negative value if the bomb occurs at the very start of the file 
                    write_start = max(start_index1 - (1*new_sample_rate), 4000)
                    write_end = max(start_index1 + (3*new_sample_rate), 44000)
                    audio_write = audio[write_start:write_end]

                    # write a new wav file for this window
                    file_counter +=1 # store the number
                    hmmss = hmmss.replace(':', '.') # remove ':' as not allowed in filenames
                    sf.write(output_folder+"\\"+str(file_counter).zfill(7)+"_"+hmmss+"_"+f[:-4]+".wav", audio_write, new_sample_rate)  

                    # note file and timepoint
                    suspected_bombs.append((f, timestamp_str1))

                # if stream 1 is not a bomb, but stream 2 is
                elif result2 == True:
                    # get timestamp of this window 
                    timestamp2 = start_index2/new_sample_rate
                    timestamp_str2 = str(datetime.timedelta(seconds=timestamp2))
                    hmmss = timestamp_str2[0:7]  # remove milliseconds
                    print('Suspected bomb at: ' + hmmss)

                    # select the 5 second window to save
                    # max is used to ensure the window isnt set to a negative value if the bomb occurs  at the very start of the file 
                    write_start = max(start_index2 - (1*new_sample_rate), 4000)
                    write_end = max(start_index2 + (3*new_sample_rate), 44000)
                    audio_write = audio[write_start:write_end]

                    # write anew wav file for this window
                    file_counter +=1 # store the number
                    hmmss = hmmss.replace(':', '.') # remove ':' as not allowed in filenames
                    sf.write(output_folder+"\\"+str(file_counter).zfill(7)+"_"+hmmss+"_"+f[:-4]+".wav", audio_write, new_sample_rate)

                    # note file and timepoint
                    suspected_bombs.append((f, timestamp_str2))


        except Exception as e:
            print(f"Error processing file {f}: {e}")
            skipped_files.append(f)
            suspected_bombs.append((f, 'file was skipped, check if corrupted'))
            continue


    # Load the existing CSV file
    existing_table = pd.read_csv(output_folder + '\\' + 'temporary_results_table.csv')

    # Create a new DataFrame with the additional data
    new_data = pd.DataFrame(suspected_bombs, columns=['File', 'Timestamp'])

    # Concatenate the existing table with the new data
    combined_table = pd.concat([existing_table, new_data])

    # Reset the index of the combined table
    combined_table.index = np.arange(1, len(combined_table) + 1)

    # Save the updated table to a CSV file without the index column
    combined_table.to_csv(output_folder + '\\' + 'temporary_results_table.csv', index=False)

    # print findings  
    #print("FILE_COUNTER: ", file_counter)
    print("\nBatch {} of {} finished. Checked {} files in this batch".format(str(batch_counter), str(total_num_batches), checked_file_counter))
    print("Found {} suspected bombs".format(len(suspected_bombs)))
    print("Skipped {} files due to corruption:".format(len(skipped_files)))
    for file in skipped_files:
        print(f"  {file}")
    print("")






if __name__ == "__main__":
    main()
