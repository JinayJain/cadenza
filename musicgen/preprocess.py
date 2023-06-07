"""
Converts the Maestro dataset into a format that can be used by the model.
"""

from enum import Enum
import functools
import os
import numpy as np
import pandas as pd
import mido
from tqdm import tqdm

DATA_DIR = "data/maestro-v3.0.0"
META_CSV = "maestro-v3.0.0.csv"


class EventType(Enum):
    NOTE_ON = 0
    NOTE_OFF = 1
    TIME_SHIFT = 2
    SET_VELOCITY = 3


EVENT_VALUE_MAX = {
    EventType.NOTE_ON: 127,
    EventType.NOTE_OFF: 127,
    EventType.TIME_SHIFT: 1,
    EventType.SET_VELOCITY: 127,
}

EVENT_TOKEN_SIZE = {
    EventType.NOTE_ON: 128,
    EventType.NOTE_OFF: 128,
    EventType.TIME_SHIFT: 125,
    EventType.SET_VELOCITY: 32,
}


@functools.cache
def get_offset(event_type: EventType):
    offset = 0

    for k, v in EVENT_TOKEN_SIZE.items():
        if k == event_type:
            break

        offset += v

    return offset


def get_event_type(token):
    offset = 0

    for k, v in EVENT_TOKEN_SIZE.items():
        if token < offset + v:
            return k

        offset += v

    raise ValueError("Token out of range")


def load_df():
    df = pd.read_csv(os.path.join(DATA_DIR, META_CSV))
    return df


def load_midi(path):
    mid = mido.MidiFile(path)
    return mid


def lerp(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def value_to_token(value, kind):
    max_value = EVENT_VALUE_MAX[kind]
    token_size = EVENT_TOKEN_SIZE[kind]

    token = int(lerp(value, 0, max_value, 0, token_size))
    token = min(token, token_size - 1)

    return token + get_offset(kind)


def token_to_value(token, kind, round=True):
    max_value = EVENT_VALUE_MAX[kind]
    token_size = EVENT_TOKEN_SIZE[kind]

    token -= get_offset(kind)

    value = lerp(token, 0, token_size - 1, 0, max_value)

    if round:
        value = int(value)

    return value


def time_to_tokens(dt):
    whole_seconds = int(dt)
    fractional_seconds = dt - whole_seconds

    tokens = []

    one_second_token = value_to_token(1, EventType.TIME_SHIFT)

    for _ in range(whole_seconds):
        tokens.append(one_second_token)

    if fractional_seconds > 0:
        tokens.append(value_to_token(fractional_seconds, EventType.TIME_SHIFT))

    return tokens


def convert_midi_to_tokens(mid: mido.MidiFile):
    dt = 0
    velocity = -1
    sustain = False
    sustained_notes = []

    tokens = []

    for msg in mid:
        dt += msg.time

        if msg.is_cc(64):  # sustain pedal
            sustain = msg.value >= 64

            # Release all sustained notes
            if not sustain and len(sustained_notes) > 0:
                tokens.extend(time_to_tokens(dt))
                dt = 0

                for note in sustained_notes:
                    tokens.append(value_to_token(note, EventType.NOTE_OFF))

                sustained_notes = []

            continue

        if msg.type not in ["note_on", "note_off"]:
            continue

        kind = msg.type if msg.velocity > 0 else "note_off"
        event_type = EventType[kind.upper()]

        if event_type == EventType.NOTE_OFF and sustain:
            sustained_notes.append(msg.note)
            continue

        tokens.extend(time_to_tokens(dt))
        dt = 0

        if event_type == EventType.NOTE_ON and velocity != msg.velocity:
            tokens.append(value_to_token(msg.velocity, EventType.SET_VELOCITY))
            velocity = msg.velocity

        tokens.append(value_to_token(msg.note, event_type))

    return tokens


def convert_tokens_to_midi(tokens) -> mido.MidiFile:
    dt = 0
    velocity = 127

    mid = mido.MidiFile()

    track = mido.MidiTrack()
    mid.tracks.append(track)

    for token in tokens:
        event_type = get_event_type(token)

        if event_type == EventType.TIME_SHIFT:
            dt += token_to_value(token, event_type, round=False)
        elif event_type == EventType.SET_VELOCITY:
            velocity = token_to_value(token, event_type)
        else:
            kind = "note_on" if event_type == EventType.NOTE_ON else "note_off"
            track.append(
                mido.Message(
                    kind,
                    note=token_to_value(token, event_type),
                    velocity=velocity,
                    time=int(
                        mido.second2tick(dt, mid.ticks_per_beat, mido.bpm2tempo(120))
                    ),
                )
            )
            dt = 0

    return mid


def main():
    df = load_df()

    midi_tokens = [
        np.array(convert_midi_to_tokens(load_midi(os.path.join(DATA_DIR, path))))
        for path in tqdm(df["midi_filename"])
    ]

    df["tokens"] = midi_tokens

    df.to_pickle("data/dataset.pkl")


if __name__ == "__main__":
    main()
