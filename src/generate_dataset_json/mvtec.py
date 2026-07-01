import argparse
import json
import os
import sys
from pathlib import Path


class MVTecSolver(object):
    CLSNAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                local_id = 0
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            local_id=local_id,
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
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
        root = Path(dataset_root) / "mvtec_anomaly_detection"
        source = "DATASET_ROOT"
    elif cli_root:
        root = Path(cli_root)
        source = "--root"
    else:
        root = repo_root.parent / "datasets" / "mvtec_anomaly_detection"
        source = "default"

    root = root.expanduser().resolve()
    print(f"MVTec root ({source}): {root}", flush=True)

    if not root.exists():
        raise FileNotFoundError(
            f"MVTec dataset root does not exist: {root}. "
            "Set DATASET_ROOT to the parent datasets directory or pass --root."
        )
    return str(root)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MVTec AD meta.json.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Path to mvtec_anomaly_detection. Ignored when DATASET_ROOT is set, "
            "because DATASET_ROOT has higher priority."
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

    runner = MVTecSolver(root=root)
    runner.run()
