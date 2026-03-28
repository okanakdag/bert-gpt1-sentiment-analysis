
import random
import numpy as np
import torch


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def load_model(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def get_unique_output_paths(metrics_path, model_path):
    """
    Return output paths that do not overwrite an existing run.

    If the base filenames already exist, this function adds a numeric suffix
    like `_2`, `_3`, and so on to both files so they stay matched.
    """
    if not metrics_path.exists() and not model_path.exists():
        return metrics_path, model_path

    suffix_index = 2

    while True:
        candidate_metrics_path = metrics_path.with_name(
            f"{metrics_path.stem}_{suffix_index}{metrics_path.suffix}"
        )
        candidate_model_path = model_path.with_name(
            f"{model_path.stem}_{suffix_index}{model_path.suffix}"
        )

        if not candidate_metrics_path.exists() and not candidate_model_path.exists():
            return candidate_metrics_path, candidate_model_path

        suffix_index += 1
