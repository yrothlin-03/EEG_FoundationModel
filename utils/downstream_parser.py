import argparse

import yaml


def _load(path):
    if path is None:
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _cast(value):
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        try:
            return float(parsed)
        except ValueError:
            return parsed
    return parsed


def _flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{key}."))
        else:
            flat[key] = v
    return flat


def _set_nested(config, dotted, value):
    keys = dotted.split(".")
    node = config
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _config_path_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_config", type=str, default="configs/model.yaml")
    parser.add_argument("--data_config", type=str, default="configs/downstream_data.yaml")
    parser.add_argument("--downstream_config", type=str, default="configs/downstream.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    return parser


def get_parser(argv=None):
    known, _ = _config_path_parser().parse_known_args(argv)

    model_config = _load(known.model_config)
    data_config = _load(known.data_config)
    downstream_config = _load(known.downstream_config)

    if known.dataset is not None:
        downstream_config["dataset"] = known.dataset
    downstream_config.setdefault("data", {}).update(data_config.get("data", {}))
    downstream_config["dataset_cfg"] = data_config[downstream_config["dataset"]]

    routing = {}
    for key in _flatten(model_config):
        routing[key] = "model"
    for key in _flatten(downstream_config):
        if key == "dataset":
            continue
        routing[key] = "downstream"

    parser = argparse.ArgumentParser(
        parents=[_config_path_parser()], description="EEG downstream training"
    )
    flat_values = {**_flatten(model_config), **_flatten(downstream_config)}
    for key, val in flat_values.items():
        if key == "dataset":
            continue
        parser.add_argument(
            f"--{key}", default=argparse.SUPPRESS, type=_cast, metavar="VALUE",
            help=f"(yaml default: {val!r})",
        )

    return parser, model_config, downstream_config, routing


def parse_config(argv=None):
    parser, model_config, downstream_config, routing = get_parser(argv)
    args = vars(parser.parse_args(argv))

    targets = {"model": model_config, "downstream": downstream_config}
    for key, source in routing.items():
        if key in args:
            _set_nested(targets[source], key, args[key])

    return model_config, downstream_config
