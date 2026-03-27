
EPOCH_DURATION = 30.

def get_event_ids():
    event_id = {'Sleep stage W': 0,
                'Sleep stage 1': 1,
                'Sleep stage 2': 2,
                'Sleep stage 3/4': 3,
                'Sleep stage R': 4}
    return event_id

def get_mapping():
    mapping = {'EOG horizontal': 'eog',
            'Resp oro-nasal': 'misc',
            'EMG submental': 'misc',
            'Temp rectal': 'misc',
            'Event marker': 'misc'}
    
    return mapping

def get_epoch_duration_const():
    return EPOCH_DURATION

def get_train_subj():
    return 0

def get_train_recording():
    return 1

def get_predict_recording():
    return 2