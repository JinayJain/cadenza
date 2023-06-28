from argparse import ArgumentParser

from tqdm import tqdm
from cadenza.data.preprocess import (
    convert_midi_to_tokens,
    convert_tokens_to_midi,
    load_midi,
)
from cadenza.model.v2.transformer import (
    CadenzaTransformer,
    CadenzaTransformerConfig,
)
from cadenza.constants import Constants
import torch


def main():
    args = parse_args()

    cuda = torch.device("cuda")
    cpu = torch.device("cpu")
    device = cpu
    model = CadenzaTransformer.load_from_checkpoint(args.model).to(cuda)

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

    # hidden = model.init_hidden(1, device)
    # _, hidden = model(tokens, hidden)

    block_size = model.cfg.block_size // 2
    model.eval()
    for i in tqdm(range(args.num_generate)):
        model_input = tokens[:, -block_size:].to(cuda)
        output = model(model_input).to(cpu)

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
