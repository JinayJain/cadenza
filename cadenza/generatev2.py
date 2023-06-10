from argparse import ArgumentParser
from cadenza.data.preprocess import (
    convert_midi_to_tokens,
    convert_tokens_to_midi,
    load_midi,
)
from cadenza.model.v2.rnn import CadenzaRNN
from cadenza.constants import Constants
import torch


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CadenzaRNN.load_from_checkpoint(args.model)

    if args.prompt is not None:
        midi = load_midi(args.prompt)
        tokens = convert_midi_to_tokens(midi)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        if args.num_prompt_tokens is not None:
            tokens = tokens[:, : args.num_prompt_tokens]
    else:
        tokens = torch.randint(
            Constants.NUM_TOKENS,
            (1, 1),
            dtype=torch.long,
            device=device,
        )

    hidden = model.init_hidden(1, device)
    _, hidden = model(tokens, hidden)

    for i in range(10_000):
        output, hidden = model(tokens[:, -1].unsqueeze(0), hidden)

        # sample from the output distribution
        token = torch.multinomial(
            torch.softmax(output[:, -1] / args.temperature, dim=-1),
            num_samples=1,
        )

        tokens = torch.cat([tokens, token], dim=1)

    midi = convert_tokens_to_midi(tokens[0])

    midi.save("sample.mid")


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output MIDI file",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Path to prompt MIDI file",
    )

    parser.add_argument(
        "--num-prompt-tokens",
        type=int,
        default=1024,
        help="Number of tokens to use from prompt",
    )

    parser.add_argument(
        "-n",
        "--num-generate",
        type=int,
        default=1024,
        help="Number of tokens to generate",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
