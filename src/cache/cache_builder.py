import logging
import os
import random
from typing import Any, Union

import numpy as np
import torch
from tqdm import tqdm

from configs import DataCfg, ModelCfg
from encoders.base_encoder import BaseEncoder

from .cache_manager import CacheManager

log = logging.getLogger(__name__)


@torch.no_grad()  # type: ignore
def build_feature_cache(
    encoder: BaseEncoder,
    device: Union[str, torch.device],
    cfg_data: DataCfg,
    cfg_model: ModelCfg,
    cache_mgr: CacheManager,
) -> None:
    torch_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng_state = np.random.get_state()
    python_rng_state = random.getstate()

    log.info("⚙️  Building feature cache ...")
    try:
        encoder.to(device).eval()

        from data import DataManager  # delay import to avoid circular import

        ds_mgr = DataManager(cfg_data, cfg_model, mode="image", batch_size=1)
        loader = ds_mgr.get_dataloader(shuffle=False, num_workers=0)

        for inputs in tqdm(loader):
            idx = inputs.global_id.item()
            save_path = cache_mgr.path_for(idx)
            if os.path.exists(save_path):
                continue

            img = inputs.img
            image_features, patch_features_list = encoder.encode_image(
                image=img,
                features_list=cfg_model.features_list,
            )
            tensor_dict = dict[str, Any](
                id=idx,
                feats=image_features.cpu(),
                mid_feats=[p.cpu() for p in patch_features_list],
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(tensor_dict, save_path, _use_new_zipfile_serialization=False)
    finally:
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        np.random.set_state(numpy_rng_state)
        random.setstate(python_rng_state)
