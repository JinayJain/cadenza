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

    prompt = None

    if args.prompt is not None:
        midi = load_midi(args.prompt)
        tokens = convert_midi_to_tokens(midi)
        prompt = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        if args.num_prompt_tokens is not None:
            prompt = prompt[:, : args.num_prompt_tokens]

    model.eval()

    generated = model.generate(
        prompt=prompt,
        num_tokens=args.num_generate,
        top_p=args.top_p,
        temperature=args.temperature,
        show_progress=True,
    )

    midi = convert_tokens_to_midi(generated[0])

    midi.save(args.output)


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

    parser.add_argument(
        "-p",
        "--top-p",
        type=float,
        default=0.9,
        help="Top-P sampling parameter",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
