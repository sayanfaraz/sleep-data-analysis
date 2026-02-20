import mne
import numpy as np

from mne.datasets.sleep_physionet.age import fetch_data

from src import consts

def load_raw_recording(subjects: list,
                          recording: list,
                          mapping: dict):
    [subject_files] = fetch_data(subjects, recording)

    raw = mne.io.read_raw_edf(subject_files[0])
    annot = mne.read_annotations(subject_files[1])

    # Keep last 100-min wake events before sleep and first 50-min wake events after
    # sleep and redefine annotations on raw data
    annot.crop(annot[1]["onset"] - 100 * 60, annot[-2]["onset"] + 50 * 60)

    raw.set_annotations(annot, emit_warning=False)
    raw.set_channel_types(mapping)

    return raw, annot

def epochs_from_recording(raw,
                          annot,
                          input_event_id: dict,
                          out_event_id: dict,
                          epoch_duration: float = 30.
    ):

        events, _ = mne.events_from_annotations(
            raw, event_id=input_event_id, chunk_duration=epoch_duration)
        
        tmax = epoch_duration - 1. / raw.info['sfreq']  # tmax in included

        epochs = mne.Epochs(raw=raw, events=events,
                                event_id=out_event_id, tmin=0., tmax=tmax, baseline=None)
        
        return epochs, events

def inds(start, duration, sfreq):
    s = int(sfreq*start)
    e = int(s+sfreq*duration)

    return s, e

def get_bands():
    """
        Return a dictionary of frequency bands and their labels in the form of (low, high].
    """
    return {
        'delta': (0, 4),
        'theta': (4, 7),
        'alpha': (7, 12),
        'beta': (12, 30),
        'gamma': (30, 50)
    }

# ---------------------------------- REGULAR

def detrend(data, sfreq):
    t = np.arange(0.0, data.shape[0])/sfreq

    trend_coeffs = np.polyfit(t, data, 1)
    trendline = t*trend_coeffs[0] + trend_coeffs[1]

    return data - trendline

def hann(data):
    return data*(1.0-np.cos(2*np.pi*np.arange(0, data.shape[0])/data.shape[0]))

def power_spec(data, sfreq, window=True, one_sided=True):
    data_preproc = hann(detrend(data, sfreq)) if window else detrend(data, sfreq)
    
    fourier = np.fft.fft(data_preproc)
    fourier_ = np.conj(fourier)
    
    dt = 1.0/sfreq
    power_spec = fourier_*fourier * 1/data_preproc.shape[0] * dt
    freq = np.fft.fftfreq(data_preproc.shape[0], d=1.0/sfreq)
    
    if one_sided:
        return freq[(0 < freq)], power_spec[(0 < freq)]
    else:
        return freq, power_spec

def bandpower(data, sfreq, window=True, relative=True, include_total=False):
    ret_bandpower = {}
    bands = get_bands()
    freq, power = power_spec(data, sfreq, window=window, one_sided=True)

    total_power = np.abs(np.trapezoid(power, freq))
    div_total_power = total_power if relative else 1.0

    for band, freq_bounds in bands.items():
        l_ind = np.argwhere(freq>freq_bounds[0])[0, 0]
        h_ind = np.argwhere(freq<=freq_bounds[1])[-1, 0]
        bpow = np.trapezoid(power[l_ind:h_ind], freq[l_ind:h_ind])
        ret_bandpower[band] = bpow / total_power
    if include_total:
        ret_bandpower['total'] = total_power if not relative else 1.0
    return ret_bandpower

# ---------------------------------- VECTORIZED

def detrend_v(data, sfreq):
    t = np.arange(0.0, data.shape[0])/sfreq

    trend_coeffs = np.polyfit(t, data, 1)
    trendline = t[:,np.newaxis]*trend_coeffs[0].T + trend_coeffs[1, np.newaxis]

    return data - trendline

def hann_v(data):
    return data*(1.0-np.cos(2*np.pi*np.arange(0, data.shape[0])/data.shape[0]))[:, np.newaxis]

def power_spec_v(data, sfreq, window=True, one_sided=True):
    data_preproc = hann_v(detrend_v(data, sfreq)) if window else detrend_v(data, sfreq)

    fourier = np.fft.fft(data_preproc, axis=0)
    fourier_ = np.conj(fourier)

#     power_spec = fourier_*fourier*(1.0/(sfreq*data_preproc.shape[0]))
    
    dt = 1.0/sfreq
    power_spec = fourier_*fourier * 1/data_preproc.shape[0] * dt
    freq = np.fft.fftfreq(data_preproc.shape[0], d=1.0/sfreq)
    
    if one_sided:
        return freq[(0 < freq)], power_spec[(0 < freq)]
    else:
#         _, power_spec = signal.periodogram(x=data_preproc, fs=sfreq, return_onesided=False, scaling='density')
        return freq, power_spec

def bandpower_v(data, sfreq, window=True, relative=True, include_total=False):
    ret_bandpower = {}
    bands = get_bands()
    freq, power = power_spec_v(data, sfreq, window=window, one_sided=True)
    # print("Freq: ", freq.shape)
    # print("Power: ", power.shape)

    total_power = np.abs(np.trapezoid(power, freq, axis=0))
    div_total_power = total_power if relative else 1.0
    # print("Total power: ", total_power.shape)

    for band, freq_bounds in bands.items():
        # print("BAND: ", band)
        l_ind = np.argwhere(freq>freq_bounds[0])[0, 0]
        h_ind = np.argwhere(freq<=freq_bounds[1])[-1, 0]
        # print("Ind: ", l_ind, h_ind)
        # print("Freq: ", freq[l_ind], freq[h_ind])
        # print("-----------------------------------")
        bpow = np.abs(np.trapezoid(power[l_ind:h_ind], freq[l_ind:h_ind], axis=0))
        ret_bandpower[band] = bpow / div_total_power
    if include_total:
        ret_bandpower['total'] = total_power if not relative else 1.0
    return ret_bandpower

def five_num_summary(data):
    # print(data)
    perc_25 = np.percentile(data, 25)
    perc_50 = np.percentile(data, 50)
    perc_75 = np.percentile(data, 75)

    iqr = perc_75 - perc_25
    perc_0 = np.min(data[data >= (perc_25 - 1.5*iqr)])
    perc_100 = np.max(data[data <= (perc_75 + 1.5*iqr)])

    num_outliers_lower = data[(data < (perc_25 - 1.5*iqr))].shape[0]
    num_outliers_upper = data[(data > (perc_75 + 1.5*iqr))].shape[0]

    return {
        "0%": perc_0,
        "25%": perc_25,
        "50%": perc_50,
        "75%": perc_75,
        "100%": perc_100,
        "# of Outliers (lower)": num_outliers_lower,
        "# of Outliers (upper)": num_outliers_upper,
    }

def bandpower_five_num_summary(data):
    ret_five_num_summary = {}
    for event in consts.get_event_id().keys():
        event_five_num_summary = {}
        for band in data[event].keys():
            event_five_num_summary[band] = five_num_summary(data[event][band])
        ret_five_num_summary[event] = event_five_num_summary
    return ret_five_num_summary