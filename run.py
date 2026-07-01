import os
import sys
from enum import Enum
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
os.environ.setdefault("REPO_ROOT", str(_repo_root))
os.environ.setdefault("DATASET_ROOT", str(_repo_root.parent / "datasets"))
os.environ.setdefault("CACHE_DIR", str(Path.home() / ".cache" / "features"))
_src = str(_repo_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
if _src not in os.environ.get("PYTHONPATH", "").split(":"):
    os.environ["PYTHONPATH"] = f"{_src}:{os.environ.get('PYTHONPATH', '')}"

import hydra
from hydra.core.config_store import ConfigStore

from configs import BaseCfg
from runners import Trainer, Evaluator, SAETrainer

cs = ConfigStore().instance()
cs.store(name="base", node=BaseCfg)

class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    TRAIN_SAE = "train_sae"

@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: BaseCfg):

    mode = Mode(cfg.mode.lower())

    print(mode)
    
    match mode:
        case Mode.TRAIN:
            Trainer(cfg).train()
        case Mode.EVAL:
            Evaluator(cfg).evaluate()
        case Mode.TRAIN_SAE:
            SAETrainer(cfg).train()
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

if __name__ == "__main__":
    main()
