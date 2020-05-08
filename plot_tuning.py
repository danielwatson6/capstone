import json
import os
import re
import sys

import matplotlib.pyplot as plt


def plot(path):

    with open(os.path.join(path, "oracle.json")) as f:
        oracle_data = json.load(f)

    hparams_schema = {}
    for hp in oracle_data.get("hyperparameters").get("space"):
        type_ = hp.get("class_name")
        config = hp.get("config")

        if type_ == "Fixed":
            continue

        if type_ in {"Choice", "Boolean"}:
            hparams_schema[config.get("name")] = {
                "type": type_,
                "choices": config.get("values") or [True, False],
            }
        else:
            hparams_schema[config.get("name")] = {
                "type": type_,
                "min": config.get("min_value"),
                "max": config.get("max_value"),
            }

    trials = [d for d in os.listdir(path) if re.match(r"^[a-z0-9]+$", d)]
    for t in trials:
        with open(os.path.join(path, t, "hparams.json")) as f:
            hparams = json.load(f)
        ...


if __name__ == "__main__":
    plot(sys.argv[1])
