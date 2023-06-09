import json

from cadenza.data.preprocess import EVENT_TOKEN_SIZE


class Constants:
    SEED = 42
    NUM_TOKENS = sum(EVENT_TOKEN_SIZE.values())
    CONTEXT_LENGTH = 512
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 100
    REPORT_INTERVAL = 100
    SAVE_INTERVAL = 10000
    CHECKPOINT_DIR = "ckpt"

    @staticmethod
    def save(path):
        with open(path, "w") as f:
            constants_dict = Constants.__dict__
            constants_dict = {
                k: v
                for k, v in constants_dict.items()
                if not k.startswith("__") and not callable(v)
            }

            json.dump(constants_dict, f, indent=4)

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            constants_dict = json.load(f)

        for k, v in constants_dict.items():
            setattr(Constants, k, v)
