from argparse import ArgumentParser
import torch
import fluidsynth
import time
import numpy as np

from musicgen.model import MusicNet
from musicgen.constants import Constants
from musicgen.preprocess import EventType, token_to_event


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = MusicNet().to(device)
    model.load_state_dict(torch.load(args.model))

    fs = fluidsynth.Synth()
    # fs.start()
    buf = []

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
    TEMPERATURE = 0.9

    start = time.time()
    total_samples = 0

    with open("live.pipe", "wb") as f:
        # for i in range(1000):
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
                    # time.sleep(event_value)

                    sample = fs.get_samples(int(event_value * 44100))
                    total_samples += len(sample)

                    elapsed_time = time.time() - start
                    elapsed_sample_time = total_samples / (44100 * 2)

                    print(
                        f"Elapsed time: {elapsed_time:.2f} s, "
                        f"Elapsed sample time: {elapsed_sample_time:.2f} s, "
                    )

                    f.write(fluidsynth.raw_audio_string(sample))

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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
