import torch
import mido
from argparse import ArgumentParser

from musicgen.model import MusicNet
from musicgen.preprocess import convert_midi_to_tokens, convert_tokens_to_midi
from musicgen.constants import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # argument parser for generate script
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the model to generate from.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The path to the MIDI file to use as a prompt.",
    )

    parser.add_argument(
        "--num-generate",
        type=int,
        default=1024,
        help="The number of tokens to generate.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="sample.mid",
        help="The path to save the generated MIDI file to.",
    )

    args = parser.parse_args()

    # load the model
    model = MusicNet().to(device)

    model.load_state_dict(torch.load(args.model))

    # load the prompt
    if args.prompt is not None:
        midi = mido.MidiFile(args.prompt)

        prompt = torch.tensor(
            convert_midi_to_tokens(midi),
            dtype=torch.long,
            device=device,
        )

    else:
        prompt = torch.randint(
            Constants.NUM_TOKENS,
            (1, 1),
            dtype=torch.long,
            device=device,
        )

    # generate the music
    model.eval()

    sample = model.generate(
        num_generate=args.num_generate,
        device=device,
        prompt=prompt,
    ).squeeze()

    # convert the tokens to MIDI
    midi = convert_tokens_to_midi(sample)

    # save the MIDI file
    midi.save(args.output)


if __name__ == "__main__":
    main()
