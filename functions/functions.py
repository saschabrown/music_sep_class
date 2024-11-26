import pandas as pd
import numpy as np
import wave
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import write
from scipy import signal
from sklearn.model_selection import train_test_split
import librosa
import tqdm as tqdm 
import random

def read_wav_file(filename):
    with wave.open(filename, 'rb') as wf:
        params = wf.getparams()
        num_channels, sampwidth, framerate, num_frames = params[:4]
        frames = wf.readframes(num_frames)
        waveform = np.frombuffer(frames, dtype=np.int16)
    return waveform, params

def read_wav_file_scipy(filename):
    framerate, waveform = wavfile.read(filename)
    return waveform, framerate

def plot_waveform(waveform, framerate):
    # Create a time array in seconds
    time_array = np.arange(0, len(waveform)) / framerate
    plt.figure(figsize=(15, 5))
    plt.plot(time_array, waveform, label="Waveform")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.legend()
    plt.show()

def read_files_in_dir(directory):
    filenames = os.listdir(directory)
    return filenames


def pick_5_samples(arrays):
    instruments = []
    for array in arrays:
        pick = np.random.choice(array, 1)
        instruments.append(pick)
    return instruments

def pick_samples_and_classify(arrays):
    #Picks a random number of samples, and returns their filepath and label
    instruments = []
    #pick at minimum two instruments
    number_of_instruments = np.random.randint(2, len(arrays) + 1)
    labels = np.zeros(len(arrays))
    already_picked = []

    while len(instruments) < number_of_instruments:
        random_pick = np.random.randint(0, len(arrays))
        if random_pick in already_picked:
            break
        else:
            already_picked.append(random_pick)
            pick = np.random.choice(arrays[random_pick], 1)
            instruments.append(pick)
            labels[random_pick] = 1

    return instruments, labels

#read the filenames, and add their data to 5 lists
def add_waveform_to_list(filenames, path = "./audio/"):
    waveforms = []
    for filename in filenames:
        waveform, params = audio_to_waveform(path + filename[0])
        waveforms.append(waveform)
    return waveforms
        
#Fast fourier transform
def fft_h(data, sample_rate):
    n = len(data)
    fft_data = np.fft.fft(data)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    return freq[:n//2], np.abs(fft_data[:n//2])

def combine_waveforms(waveforms):
    normalization = 1 / len(waveforms)
    out = np.zeros_like(waveforms[0], dtype=np.float32)
    for w in waveforms:
        out += w.astype(np.float32) * normalization
    return out # note, this retuns a float32 array - it is needed to convert this to int16 before saving it to a wav file


def waveform_to_wavfile(waveform, name_string, sample_rate = 16000):
    write(name_string, sample_rate, waveform.astype(np.int16))

def create_spectrogram(waveform, sample_rate = 16000):
    freq, ts, spectro = signal.spectrogram(waveform, sample_rate)
    return freq, ts, spectro

def invert_spectrogram(freq, spectrogram):
    return signal.istft(spectrogram, freq)

def plot_spectrogram(freq, ts, spectrogram):
    plt.figure(figsize=(15, 5))
    plt.pcolormesh(ts, freq, spectrogram, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.show()

def gen_combo_waveform(filenames):
    pianos = [filename for filename in filenames if "piano" in filename] #empty
    bass = [filename for filename in filenames if "bass" in filename]
    guitar = [filename for filename in filenames if "guitar" in filename]
    drum = [filename for filename in filenames if "drum" in filename] #empty
    flutes = [filename for filename in filenames if "flute" in filename]
    keyboards = [filename for filename in filenames if "keyboard" in filename]
    paths, label = pick_samples_and_classify([pianos, bass, guitar, drum, flutes, keyboards])
    waveforms = add_waveform_to_list(paths)
    return combine_waveforms(waveforms), label

def gen_data_set(N):
    data = []
    labels = []
    for i in range(N):
        waveform, label = gen_combo_waveform()
        
        data.append(waveform)
        labels.append(label)
    return data, labels

def gen_spectrogram_set(N):
    data = []
    labels = []
    for i in range(N):
        waveform, label = gen_combo_waveform()
        freq, ts, spectro = create_spectrogram(waveform)
        data.append(spectro)
        labels.append(label)
    return data, labels
#Sorting the files in directory

def set_file_path(filepath):
    path = filepath
    return path

def create_single_inst_classification_set(N_samples = 1000, classes = ['bass', 'flute', 'guitar', 'keyboard'], path = "./audio/"):
    filenames = read_files_in_dir(path)
    data_spectogrmas = []
    labels = []
    unbalanced = False
    unbalanced_classes = []
    available_classes = np.arange(len(classes))

    # Sort the files into the classes
    instrument_files = np.zeros(len(classes), dtype=object)
    for i in range(len(classes)):
        instrument_files[i] = [filename for filename in filenames if classes[i] in filename]

    # Check if there are enough files in the classes
    file_amounts = [len(instrument_files[i]) for i in range(len(classes))]
    total_file_amount = sum(file_amounts)

    if total_file_amount < N_samples:
        print("Not enough files in the classes")
        print("Total files: ", total_file_amount)
        return
    
    available_classes = np.arange(len(classes))
    # Generate random integers for the classes\
    random_ints = []
    counter = np.zeros(len(classes))
    for i in range(N_samples):
        random_int = int(np.random.choice(available_classes, 1))
        random_ints.append(random_int)
        counter[random_int] += 1
        if int(counter[random_int]) == int(file_amounts[random_int]):
            unbalanced = True
            if classes[random_int] not in unbalanced_classes:
                unbalanced_classes.append(classes[random_int])
            available_classes = np.delete(available_classes, np.where(available_classes == random_int))

    for i in range(N_samples):
        random_int = random_ints[i]
        # Pick a random file from the instrument and pop it from the list
        audio_file = np.random.choice(instrument_files[random_int], 1)
        # Remove the file from the list
        instrument_files[random_int] = np.delete(instrument_files[random_int], np.where(instrument_files[random_int] == audio_file[0]))
        waveform, params = read_wav_file_scipy(path + audio_file[0])
        freq, ts, spectro = create_spectrogram(waveform)
        # Flatten the spectrogram
        spectro = spectro.flatten()
        data_spectogrmas.append(spectro)
        labels.append(random_int)

    # Create a label dictionary
    label_dict = {}
    for i, label in enumerate(classes):
        label_dict[i] = label

    # Convert to numpy arrays
    data_spectogrmas = np.array(data_spectogrmas)
    labels = np.array(labels)

    if unbalanced:
        print("WARNING: not enough files in classes: ", unbalanced_classes)
        print("The dataset might be unbalanced")

    return data_spectogrmas, labels, label_dict

def create_single_inst_classification_set_new_input(N_samples = 1000, classes = ['bass', 'flute', 'guitar', 'keyboard'], path = "./audio/"):
    filenames = read_files_in_dir(path)
    data_spectograms = []
    labels = []
    unbalanced = False
    unbalanced_classes = []
    available_classes = np.arange(len(classes))

    # Sort the files into the classes
    instrument_files = np.zeros(len(classes), dtype=object)
    for i in range(len(classes)):
        instrument_files[i] = [filename for filename in filenames if classes[i] in filename]

    # Check if there are enough files in the classes
    file_amounts = [len(instrument_files[i]) for i in range(len(classes))]
    total_file_amount = sum(file_amounts)

    if total_file_amount < N_samples:
        print("Not enough files in the classes")
        print("Total files: ", total_file_amount)
        return
    
    available_classes = np.arange(len(classes))
    # Generate random integers for the classes\
    random_ints = []
    counter = np.zeros(len(classes))
    for i in range(N_samples):
        random_int = int(np.random.choice(available_classes, 1))
        random_ints.append(random_int)
        counter[random_int] += 1
        if int(counter[random_int]) == int(file_amounts[random_int]):
            unbalanced = True
            if classes[random_int] not in unbalanced_classes:
                unbalanced_classes.append(classes[random_int])
            available_classes = np.delete(available_classes, np.where(available_classes == random_int))

    for i in range(N_samples):
        random_int = random_ints[i]
        # Pick a random file from the instrument and pop it from the list
        audio_file = np.random.choice(instrument_files[random_int], 1)
        # Remove the file from the list
        instrument_files[random_int] = np.delete(instrument_files[random_int], np.where(instrument_files[random_int] == audio_file[0]))
        waveform, params = audio_to_waveform(path + audio_file[0])
        spectro = waveform_to_spectrogram(waveform)
        # Flatten the spectrogram
        spectro = spectro.flatten()
        data_spectograms.append(spectro)
        labels.append(random_int)

    # Create a label dictionary
    label_dict = {}
    for i, label in enumerate(classes):
        label_dict[i] = label

    # Convert to numpy arrays
    data_spectograms = np.array(data_spectograms)
    labels = np.array(labels)

    if unbalanced:
        print("WARNING: not enough files in classes: ", unbalanced_classes)
        print("The dataset might be unbalanced")

    return data_spectograms, labels, label_dict

def split_data(data, labels, val_frac = 0.1, test_frac = 0.1, random_state = 42):
    total_frac = val_frac + test_frac
    X_train, X_val_test, y_train, y_val_test = train_test_split(data, labels, test_size=total_frac, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_frac/total_frac, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def audio_to_waveform(audio):
    waveform, sr = librosa.load(audio, sr=None)
    return waveform, sr

def waveform_to_spectrogram(waveform):
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    spectrogram = librosa.stft(waveform)
    return np.abs(spectrogram)

def pad_spectrogram(spec, target_shape):
    padded_spec = np.zeros(target_shape,dtype = np.float32)
    min_shape = np.minimum(target_shape, spec.shape)
    padded_spec[:min_shape[0], :min_shape[1]] = spec[:min_shape[0], :min_shape[1]]
    return padded_spec

def spectrogram_to_audio(spectrogram, sr,output_wav):
    waveform = librosa.istft(spectrogram)
    waveform = waveform/np.max(np.abs(waveform))
    return write(output_wav, sr, (waveform*32767).astype(np.int16))


def normalize_spectrogram(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-6)
    return normalized_spectrogram

def find_longest_array(arrays):
    longest = 0
    for array in arrays:
        if len(array) > longest:
            longest = len(array)
    return longest



def waveform_to_spectrogram(waveform):
    waveform = waveform.astype(np.float32)
    spectrogram = librosa.stft(waveform)
    return np.abs(spectrogram)

#Changed: make sure to input list of instruments paths
def nu_gen_spectro(N, instrument_list, path = "./audio" ,target_shape=(129, 285), nperseg=2048, noverlap=512):
    data = []
    labels = []
    original_labels = []


    for i in range(N):

        paths, label = pick_samples_and_classify(instrument_list)

        original_labels.append(label)
        waveforms = add_waveform_to_list(paths, path = path)
        mixed_waveform = combine_waveforms(waveforms)
        
        mixed_spectro = waveform_to_spectrogram(mixed_waveform)
        mixed_spectro_padded = pad_spectrogram(mixed_spectro, target_shape)
        mixed_spectro_normalized = normalize_spectrogram(mixed_spectro_padded)
        
        inter_waveforms = []
        

        inst_i = 0
        for n, i in enumerate(label):
            if i == 1:
                spectro = waveform_to_spectrogram(waveforms[inst_i])
                spectro_padded = pad_spectrogram(spectro, target_shape)
                

                inter_waveforms.append(spectro_padded)
                inst_i += 1
      
            if i == 0:
                inter_waveforms.append(np.zeros(target_shape))
      

        
        data.append(mixed_spectro_normalized)
        labels.append(inter_waveforms)
    
    data = np.array(data)
    
    return data, np.array(labels), np.array(original_labels) # remove last line if you want to return only data and labels

def generate_mixed_spectrograms(n_mixed_spectrograms, number_of_instruments = 3, path = "./audio/"):
    filenames = read_files_in_dir(path)
    organ = [filename for filename in filenames if "organ" in filename] 
    bass = [filename for filename in filenames if "bass" in filename]
    guitar = [filename for filename in filenames if "guitar" in filename]
    vocal = [filename for filename in filenames if "vocal" in filename] 
    flutes = [filename for filename in filenames if "flute" in filename]
    keyboards = [filename for filename in filenames if "keyboard" in filename] 
    instruments = [organ, bass, guitar, vocal, flutes, keyboards]
    picked_inst_arr = np.zeros((n_mixed_spectrograms, len(instruments)))
    
    for i in range(n_mixed_spectrograms):
        # Set 3 random values in picked_inst_arr to 1
        random_indices = np.random.choice(len(instruments), number_of_instruments, replace=False)
        for j in random_indices:
            picked_inst_arr[i, j] = 1

    mixed_spectograms = []
    for i in tqdm.tqdm(range(n_mixed_spectrograms)):
        selected_files = []
        # Select files from the picked instruments
        for j in range(len(instruments)):
            if picked_inst_arr[i, j] == 1:
                selected_files.append(random.choice(instruments[j]))

        # Generate the mixed spectrogram, and save the individual spectrograms
        for j in range(len(selected_files)):
            waveform_test, sr = audio_to_waveform(path + selected_files[j])
            if j == 0:
                combined_waveform = waveform_test
            else:
                combined_waveform = combined_waveform + waveform_test

        mixed_spectogram = waveform_to_spectrogram(combined_waveform)
        mixed_spectograms.append(mixed_spectogram)

    # Convert to numpy arrays
    mixed_spectograms = np.array(mixed_spectograms)
    picked_inst_arr = np.array(picked_inst_arr)

    return mixed_spectograms,  picked_inst_arr

def to_mel_spectrogram(spectrogram, sr, n_mels = 128):
    return librosa.feature.melspectrogram(y = None, sr=sr, S = spectrogram, n_mels=128)


def nu_gen_mel_spectro(N, instrument_list, path = "./audio" ,target_shape=(129, 285), sr = 22050, nperseg=2048, noverlap=512):
    data = []
    labels = []
    original_labels = []


    for i in range(N):

        paths, label = pick_samples_and_classify(instrument_list)

        original_labels.append(label)
        waveforms = add_waveform_to_list(paths, path = path)
        mixed_waveform = combine_waveforms(waveforms)
        
        mixed_spectro = waveform_to_spectrogram(mixed_waveform)
        mixed_spectro_padded = pad_spectrogram(mixed_spectro, target_shape)
        mixed_spectro_normalized = normalize_spectrogram(mixed_spectro_padded)
        
        inter_waveforms = []
        

        inst_i = 0
        for n, i in enumerate(label):
            if i == 1:
                spectro = waveform_to_spectrogram(waveforms[inst_i])
                spectro_padded = pad_spectrogram(spectro, target_shape)
                

                inter_waveforms.append(to_mel_spectrogram(spectro_padded, sr))
                inst_i += 1
      
            if i == 0:
                inter_waveforms.append(to_mel_spectrogram(np.zeros(target_shape), sr))
      

        
        data.append(to_mel_spectrogram(mixed_spectro_normalized, sr))
        labels.append(inter_waveforms)
    
    data = np.array(data)
    
    return data, np.array(labels), np.array(original_labels) # remove last line if you want to return only data and labels


def pick_constant_samples_and_classify(arrays):
    # Picks a constant number of samples and returns their filepath and label
    instruments = []
    # Pick at minimum two instruments
    number_of_instruments = np.random.randint(2, len(arrays) + 1)
    labels = np.zeros(len(arrays))
    already_picked = []

    while len(instruments) < number_of_instruments:
        random_pick = np.random.randint(0, len(arrays))
        if random_pick in already_picked:
            continue  # Skip this iteration and proceed to the next iteration
        else:
            already_picked.append(random_pick)
            pick = np.random.choice(arrays[random_pick], 1)
            instruments.append(pick)
            labels[random_pick] = 1

    # Continue adding instruments until the length of instruments is 3 or greater
    while len(instruments) < 2:
        random_pick = np.random.randint(0, len(arrays))
        if random_pick in already_picked:
            continue  # Skip this iteration and proceed to the next iteration
        else:
            already_picked.append(random_pick)
            pick = np.random.choice(arrays[random_pick], 1)
            instruments.append(pick)
            labels[random_pick] = 1

    return instruments, labels
