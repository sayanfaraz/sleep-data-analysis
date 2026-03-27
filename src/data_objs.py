from src import preprocess
from src import consts

class EEGRecording:
    def __init__(self, subject, recording_no, raw, annot, epochs, events, sfreq, epoch_duration):
        self.subject = subject
        self.recording_no = recording_no

        self.raw = raw
        self.annot = annot
        self.epochs = epochs
        self.events = events

        self.sfreq = sfreq
        self.epoch_duration = epoch_duration

class SleepSubject:
    def __init__(self, subject, epoch_duration=30.):
        self._subject = subject
        self._epoch_duration = epoch_duration
        self._recordings = {}

    def get_recording(self, recording_no: int):
        if recording_no not in self._recordings:
            raw, annot = preprocess.load_raw_recording(
                subjects=[self._subject],
                recording=[recording_no],
                mapping=consts.get_mapping()
            )

            # Combining Sleep Stage 3 and 4
            annotation_desc_2_event_id = {'Sleep stage W': 0,
                                        'Sleep stage 1': 1,
                                        'Sleep stage 2': 2,
                                        'Sleep stage 3': 3,
                                        'Sleep stage 4': 3,
                                        'Sleep stage R': 4}

            epochs, events = preprocess.epochs_from_recording(
                raw,
                annot,
                input_event_id=annotation_desc_2_event_id,
                out_event_id=consts.get_event_ids(),
                epoch_duration=self._epoch_duration
            )

            self._recordings[recording_no] = EEGRecording(
                self._subject, recording_no,
                raw, annot, epochs, events,
                raw.info['sfreq'],
                self._epoch_duration
            )

        return self._recordings[recording_no]