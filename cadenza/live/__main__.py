from argparse import ArgumentParser
import torch
import fluidsynth
import time
import numpy as np

from cadenza.model.rnn import CadenzaRNN
from cadenza.constants import Constants
from cadenza.data.preprocess import EventType, token_to_event


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = CadenzaRNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    fs = fluidsynth.Synth()
    if not args.output:
        fs.start()

    sfid = fs.sfload(args.soundfont)

    fs.program_select(0, sfid, 0, 0)

    x = torch.randint(
        Constants.NUM_TOKENS,
        (1, 1),
        dtype=torch.long,
        device=device,
    )
    hidden = model.init_hidden(1, device)

    velocity = 127
    TEMPERATURE = args.temperature

    if args.output:
        f = open(args.output, "wb")

    while True:
        x, hidden = model(x, hidden)

        x = x.squeeze().div(TEMPERATURE).exp()
        x = torch.multinomial(x, num_samples=1).unsqueeze(0)

        token = x.item()
        event_type, event_value = token_to_event(token)

        match event_type:
            case EventType.NOTE_ON:
                fs.noteon(0, event_value, velocity)
            case EventType.NOTE_OFF:
                fs.noteoff(0, event_value)
            case EventType.TIME_SHIFT:
                if args.output:
                    sample = fs.get_samples(int(event_value * 44100))
                    f.write(fluidsynth.raw_audio_string(sample))
                else:
                    time.sleep(event_value)
            case EventType.SET_VELOCITY:
                velocity = event_value

    fs.delete()


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The path to the model to load.",
    )

    parser.add_argument(
        "-s",
        "--soundfont",
        type=str,
        required=True,
        help="The path to the soundfont to use.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to the pipe to write to.",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature to use when sampling.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
