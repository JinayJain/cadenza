import json


class Constants:
    NUM_TOKENS = 128 + 128 + 125 + 32
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    DROPOUT = 0.2
    SEQ_LENGTH = 512
    BATCH_SIZE = 32
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
