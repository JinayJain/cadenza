from argparse import ArgumentParser
import asyncio
import time
import fluidsynth
import torch
from cadenza.constants import Constants
from cadenza.data.preprocess import EventType, token_to_event

from cadenza.model.v2.transformer import CadenzaTransformer


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
        "-f",
        "--soundfont",
        type=str,
        required=True,
        help="Path to soundfont",
    )

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=Constants.NUM_TOKENS,
        help="Number of top-k tokens to sample from",
    )

    return parser.parse_args()


async def generator(ckpt_path: str, k: int, event_queue: asyncio.Queue):
    with torch.no_grad():
        print("Generator starting...")
        model = CadenzaTransformer.load_from_checkpoint(ckpt_path)

        model.eval()

        tokens = torch.randint(
            Constants.NUM_TOKENS,
            (1, 1),
            dtype=torch.long,
        ).to(model.device)

        while True:
            tokens = tokens[:, -model.cfg.block_size :]

            output = model(tokens)

            # token = torch.multinomial(
            #     torch.softmax(output[:, -1] / temperature, dim=-1),
            #     num_samples=1,
            # )

            top_k = torch.topk(output[:, -1], k=k, dim=-1)

            token_idx = torch.multinomial(
                torch.softmax(top_k.values, dim=-1),
                num_samples=1,
            )

            token = top_k.indices[0, token_idx]

            tokens = torch.cat([tokens, token], dim=1)

            event = token_to_event(token[0].item())

            await event_queue.put(event)


async def main():
    args = parse_args()

    fs = fluidsynth.Synth()
    fs.start()

    sfid = fs.sfload(args.soundfont)
    fs.program_select(0, sfid, 0, 0)

    velocity = 127

    event_queue = asyncio.Queue(256)

    asyncio.create_task(generator(args.model, args.top_k, event_queue))

    while True:
        kind, value = await event_queue.get()

        if event_queue.qsize() < event_queue.maxsize // 3:
            print("Generator queue is running low!")

        match kind:
            case EventType.NOTE_ON:
                fs.noteon(0, value, velocity)
            case EventType.NOTE_OFF:
                fs.noteoff(0, value)
            case EventType.TIME_SHIFT:
                await asyncio.sleep(value)
            case EventType.SET_VELOCITY:
                velocity = value


if __name__ == "__main__":
    asyncio.run(main())
