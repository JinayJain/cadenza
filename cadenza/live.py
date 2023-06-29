from argparse import ArgumentParser
import asyncio
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
        "-p",
        "--top-p",
        type=float,
        default=0.9,
        help="Top-P sampling parameter",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )

    parser.add_argument(
        "-q",
        "--queue-size",
        type=int,
        default=256,
        help="Size of event queue",
    )

    return parser.parse_args()


async def generator(
    ckpt_path: str,
    p: float,
    temperature: float,
    event_queue: asyncio.Queue,
):
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
            token = model.generate_single(
                tokens[:, -Constants.CONTEXT_LENGTH :],
                top_p=p,
                temperature=temperature,
            )
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

    event_queue = asyncio.Queue(maxsize=args.queue_size)

    asyncio.create_task(
        generator(args.model, args.top_p, args.temperature, event_queue)
    )

    while True:
        kind, value = await event_queue.get()

        if event_queue.qsize() < event_queue.maxsize // 3:
            print("Generator queue is running low!")

        if kind == EventType.NOTE_ON:
            fs.noteon(0, value, velocity)
        elif kind == EventType.NOTE_OFF:
            fs.noteoff(0, value)
        elif kind == EventType.TIME_SHIFT:
            await asyncio.sleep(value)
        elif kind == EventType.SET_VELOCITY:
            velocity = value


if __name__ == "__main__":
    asyncio.run(main())
