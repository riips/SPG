import logging
import os
from typing import Any, Optional

import torch

log = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, save_dir: str, device: Optional[str] = None):
        self.save_dir = save_dir
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def save(self, epoch: int, **components: Any) -> None:
        """
        Save the checkpoint.
        """

        state = {"epoch": epoch + 1}
        for name, component in components.items():
            if hasattr(component, "state_dict"):
                component = component.state_dict()
                key = f"{name}_state_dict"
            else:
                key = name
            state[key] = component

        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"epoch_{epoch+1}.pth")
        torch.save(state, path)
        log.info(f"Checkpoint saved: {path}")

    def load(self, path: str, **components: Any) -> tuple[int, dict[str, Any]]:
        states = torch.load(path, map_location=self.device)
        loaded_components: dict[str, Any] = {}

        for name, component in components.items():
            state_dict_key = f"{name}_state_dict"

            if state_dict_key in states:
                # Load components that expose .state_dict().
                if hasattr(component, "load_state_dict"):
                    # missing, unexpected = component.load_state_dict(states[state_dict_key], strict=False)

                    ######### Backward compatibility: replace guide. with guide_0. #########
                    sd = states[state_dict_key]
                    if isinstance(sd, dict) and "guide._guide_tensors" in sd and "guide_0._guide_tensors" not in sd:
                        sd = {k.replace("guide.", "guide_0.", 1): v for k, v in sd.items()}
                    missing, unexpected = component.load_state_dict(sd, strict=False)
                    ###############################################################

                    if missing or unexpected:
                        log.warning("Component load_state_dict mismatch: missing=%s unexpected=%s", missing, unexpected)
                    loaded_components[name] = component
                else:
                    log.warning(
                        f"Warning: component '{name}' does not have 'load_state_dict' method."
                    )
            elif name in states:
                # Other objects, such as pickled objects.
                loaded_components[name] = states[name]
            else:
                log.warning(f"Warning: '{name}' not found in checkpoint.")

        epoch = states.get("epoch", 0)
        log.info(f"Checkpoint loaded: {path}, epoch: {epoch}")
        return epoch, loaded_components
