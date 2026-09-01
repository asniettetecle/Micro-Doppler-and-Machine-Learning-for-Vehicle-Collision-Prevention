#Load relevant libraries
import json, numpy, matplotlib.pyplot
from scipy.signal import stft, find_peaks
from utility.mmw_cube_proc_v0 import CubeProcessor
from openpyxl import Workbook, load_workbook

excel_table = r"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\ExperimentalRadarDataFileNames - Augmented.xlsx"

workbook_excel = load_workbook(excel_table, data_only = True)

table = []
for row in workbook_excel.active.iter_rows(values_only = True):
    table.append(list(row))

header = table[0]
data = table[1:]

dictionary = {}
for row in data:
    FileName, Subject, TargetClass, Activity, Angle, Repetition, AugmentedRepetition = row
    dictionary[FileName] = (Subject, TargetClass, Activity, Angle, Repetition, AugmentedRepetition)

for FileName, (Subject, TargetClass, Activity, Angle, Repetition, SimulatedRepetition) in dictionary.items():

    data_file = rf"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\ExperimentalRadarData - Augmented\{FileName}.bin"

    radar_configuration = r"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\radar_config\config_3rx_3m\BGT60TR13C_settings_20241101-104314.json"

    with open(radar_configuration, "r") as file:
        setting = json.load(file)
    processed_cube = CubeProcessor(setting, mti_alpha = 1, num_doppler_bin = 16)

    #Doppler-Time
    doppler_time = []
    file = open(data_file, "rb")    #Open data file
    while file.read(4):    #Loop through file time frame by time frame
        file.read(4)   #Skip first 4 bytes (sequence number)
        raw_data_length = int.from_bytes(file.read(4), "little")    #Read next 4 bytes left to right as integer & store result (raw data length)
        raw_ADC_data = file.read(raw_data_length)        #Read following bytes (number of bytes = raw_data_length) (raw ADC data)
        processed_cube.process_raw_data(raw_ADC_data)   #Convert raw radar signal data to 4D cube of signals containing Doppler, range, azimuth & elevation info
        doppler_time.append(numpy.mean(numpy.sum(processed_cube.data_cube_fft, axis = 1), axis = (1,2)))
        #data_cube_fft = signal cube containing (Doppler, Range, Azimuth, Elevation) info
        #numpy.sum(processed_cube.data_cube_fft, axis = 1) -> apply FFT across range bins, sum range bins & collapse from signal data cube, signal data cube = (Doppler, Azimuth, Elevation)
        #numpy.mean(numpy.sum(processed_cube.data_cube_fft, axis = 1), axis = (1,2)) -> takes average of azimuth & elevation info & collapse from signal data cube, gets 1D array of complex/raw signal strengths for 1 Doppler bin 
        #doppler_time.append(numpy.mean(numpy.sum(processed_cube.data_cube_fft, axis = 1), axis = (1,2))) -> adds each term in 1D array of raw/complex signal strengths per Doppler bin to doppler_time array 
        #AT THIS POINT doppler_time is an array of mini-arrays, each array showing all Doppler bin signals in 1 TIME FRAME
    doppler_time = numpy.stack(doppler_time, axis = 1)   #doppler_time array goes from 1D to 2D array, each row now is 1 DOPPLER BIN SIGNAL across all times

    #STFT across time for each Doppler bin
    frame_repetition_time = 0.2   #Time between each radar frame (s) (aka duration of a chirp)
    fs = 1 / frame_repetition_time   #Sampling frequency, how many frames/samples happen per second
    time_window = 2   #How many time frames looked at at once (in 1 STFT window)
    shared_frames = time_window * 0.5  #Number of shared/overlapping time frames in consecutive STFT windows
    num_doppler_bins = doppler_time.shape[0]   #Total number of Doppler bins
    stft_per_doppler_bin = []

    #Find STFT for each Doppler bin
    for i in range(num_doppler_bins):
        signal = doppler_time[i, :]   #Looks at 1 Doppler bin over time
        f, t, STFT = stft(signal, fs = fs, nperseg = time_window, noverlap = shared_frames, boundary = None, padded = 0)   #STFT: rows = frequencies, columns = time windows, each value = complex FFT over time window (transformed signal) 
        stft_per_doppler_bin.append(10*numpy.log10(numpy.abs(STFT).mean(axis = 0)))   #stft_per_doppler_bin = 1D array of signal strengths (dB) FOR 1 DOPPLER BIN, store signal strength (in dB) (and frequency (0) averaged & collapsed)

    #full_stft = 2D array of singal strengths for each Doppler bin & time window
    full_stft = numpy.array(stft_per_doppler_bin)   #full_stft: row = signal strength (dB) for Doppler bin, column = signal strength (dB) for time window 

    #X-axis = time
    time = t
    #Y-axis = Doppler frequencies
    doppler_frequencies = processed_cube.proc_param["doppler_bin"]

    #Plot Doppler Frequency – Time Spectrogram
    matplotlib.pyplot.figure()
    im = matplotlib.pyplot.imshow(full_stft, aspect = "auto", origin = "lower", extent = [time[0], time[-1], doppler_frequencies[0], doppler_frequencies[-1]], cmap = "jet")
    matplotlib.pyplot.xlabel("Time (s)", fontsize=16)
    matplotlib.pyplot.ylabel("Doppler Frequency (Hz)", fontsize=16)
    matplotlib.pyplot.title(f"Doppler – Time Spectrogram:\n{FileName}", fontsize=18)
    cbar = matplotlib.pyplot.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Signal Strength (dB)", fontsize=16)
    matplotlib.pyplot.tick_params(axis="both", labelsize=14)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(rf"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\Doppler-Time STFTs - Augmented\Doppler-Time STFT - {FileName}.png")
    matplotlib.pyplot.close()

    #Plain spectrograms for ML modelling (no title, axis, labelling)
    #Only save spectrograms for selected activities
    target_classes = ["Cyclist", "Pedestrian"]

    if TargetClass in target_classes:

        matplotlib.pyplot.figure()
        matplotlib.pyplot.imshow(full_stft, aspect = "auto", origin = "lower", cmap = "jet")
        matplotlib.pyplot.axis("off")
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(rf"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\Doppler-Time STFTs - Augmented - ML\{TargetClass}\{FileName}.png")
        matplotlib.pyplot.close()

    min_doppler_frequency = []
    max_doppler_frequency = []
    doppler_bandwidth = []
    peak_signal_strength = []
    standard_deviation_signal = []

    for i in range(len(time)):
        dominant_signal = doppler_frequencies[full_stft[:, i] > (full_stft[:, i].min() + 0.6 * (full_stft[:, i].max() - full_stft[:, i].min()))]
        min_doppler_frequency.append(dominant_signal.min())
        max_doppler_frequency.append(dominant_signal.max())
        doppler_bandwidth.append(dominant_signal.max() - dominant_signal.min())
        peak_signal = numpy.max(full_stft, axis = 0)
        peak_signal_strength.append(peak_signal[i])
        standard_deviation = numpy.std(full_stft, axis = 0)
        standard_deviation_signal.append(standard_deviation[i])

    #DESIRED FEATURES
        #Min of standard_deviation_signal
        #STD of standard_deviation_signal
        #Max of standard_deviation_signal
        #STD of min_doppler_frequency
        #STD of peak_signal_strength
        #STD of max_doppler_frequency
        #Max of peak_signal_strength
        #STD of doppler_bandwidth
        #Mean of peak_signal_strength
        #Mean of max_doppler_frequency

    feature_vector = []

    min_doppler_frequency_features = numpy.asarray(min_doppler_frequency)
    max_doppler_frequency_features = numpy.asarray(max_doppler_frequency)
    doppler_bandwidth_features = numpy.asarray(doppler_bandwidth)
    peak_signal_strength_features = numpy.asarray(peak_signal_strength)
    standard_deviation_signal_features = numpy.asarray(standard_deviation_signal)

    feature_vector.append(standard_deviation_signal_features.min())
    feature_vector.append(standard_deviation_signal_features.std())
    feature_vector.append(standard_deviation_signal_features.max())
    feature_vector.append(min_doppler_frequency_features.std())
    feature_vector.append(peak_signal_strength_features.std())
    feature_vector.append(max_doppler_frequency_features.std())
    feature_vector.append(peak_signal_strength_features.max())
    feature_vector.append(doppler_bandwidth_features.std())
    feature_vector.append(peak_signal_strength_features.mean())
    feature_vector.append(max_doppler_frequency_features.mean())

    workbook = Workbook()
    workbook.active.title = "Features"
    workbook.active.append(["Time (s)", "Min Doppler (Hz)", "Max Doppler (Hz)", "Doppler Bandwidth (Hz)", "Peak Signal Strength (dB)", "STD of Signal (dB)"])
    for i in range(len(time)):
        workbook.active.append([float(time[i]), float(min_doppler_frequency[i]), float(max_doppler_frequency[i]), float(doppler_bandwidth[i]), float(peak_signal_strength[i]), float(standard_deviation_signal[i])])

    feature_vector_sheet = workbook.create_sheet(TargetClass)

    feature_titles = ["Min of STD of Signal (dB)",
                      "STD of STD of Signal (dB)",
                      "Max of STD of Signal (dB)",
                      "STD of Min Doppler (Hz)",
                      "STD of Peak Signal Strength (dB)",
                      "STD of Max Doppler (Hz)",
                      "Max of Peak Signal Strength (dB)",
                      "STD of Bandwidth (Hz)",
                      "Mean of Peak Signal Strength (dB)",
                      "Mean of Max Doppler (Hz)"]
    
    for i, j in zip(feature_titles, feature_vector):
        feature_vector_sheet.append([i, j])

    workbook.save(rf"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\Features - Augmented\Features - Doppler-Time STFT - {FileName}.xlsx")
