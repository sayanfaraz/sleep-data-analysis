import mne

from mne.datasets.sleep_physionet.age import fetch_data

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