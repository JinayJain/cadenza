import torch
import mido
from argparse import ArgumentParser

from cadenza.model.rnn import CadenzaRNN
from cadenza.data.preprocess import convert_midi_to_tokens, convert_tokens_to_midi
from cadenza.constants import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # argument parser for generate script
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The path to the model to generate from.",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="The path to the MIDI file to use as a prompt.",
    )

    parser.add_argument(
        "-n",
        "--num-generate",
        type=int,
        default=1024,
        help="The number of tokens to generate.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="sample.mid",
        help="The path to save the generated MIDI file to.",
    )

    args = parser.parse_args()

    # load the model
    model = CadenzaRNN().to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))

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
