import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd


class VisASolver(object):
    CLSNAMES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum',
    ]

    def __init__(self, root='data/visa'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']
        self.csv_data = pd.read_csv(f'{root}/split_csv/1cls.csv', header=0)

    def run(self):
        columns = self.csv_data.columns  # [object, split, label, image, mask]
        info = {phase: {} for phase in self.phases}
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_data = self.csv_data[self.csv_data[columns[0]] == cls_name]
            for phase in self.phases:
                cls_info = []
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                cls_data_phase.index = list(range(len(cls_data_phase)))
                local_id = 0
                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.loc[idx]
                    is_abnormal = True if data[2] == 'anomaly' else False
                    info_img = dict(
                        local_id = local_id,
                        img_path=data[3],
                        mask_path=data[4] if is_abnormal else '',
                        cls_name=cls_name,
                        specie_name='',
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
                    local_id += 1
                    if phase == 'test':
                        if is_abnormal:
                            anomaly_samples = anomaly_samples + 1
                        else:
                            normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)


def resolve_root(cli_root=None):
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = os.environ.get("DATASET_ROOT")

    if dataset_root:
        root = Path(dataset_root) / "visa"
        source = "DATASET_ROOT"
    elif cli_root:
        root = Path(cli_root)
        source = "--root"
    else:
        root = repo_root.parent / "datasets" / "visa"
        source = "default"

    root = root.expanduser().resolve()
    print(f"VisA root ({source}): {root}", flush=True)

    if not root.exists():
        raise FileNotFoundError(
            f"VisA dataset root does not exist: {root}. "
            "Set DATASET_ROOT to the parent datasets directory or pass --root."
        )
    return str(root)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VisA meta.json.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Path to visa. Ignored when DATASET_ROOT is set, because DATASET_ROOT "
            "has higher priority."
        ),
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        root = resolve_root(args.root)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    runner = VisASolver(root=root)
    runner.run()
