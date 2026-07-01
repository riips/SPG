import torch

from configs import ModelCfg

from . import CLIP
from .base_encoder import BaseEncoder

class OpenCLIP(BaseEncoder):
    def __init__(self, encoder_name: str, device: str | torch.device, model_cfg: ModelCfg):
        super(OpenCLIP, self).__init__(encoder_name, device)
        self.cfg = model_cfg
        self._load_model()

    def _load_model(self):
        self.model, _ = CLIP.load(
            self.cfg.id,
            device=self.device, 
            download_root=self.cfg.root
        )
        self.visual = self.model.visual
        self.output_dim = self.model.visual.output_dim
    
    def encode_image(self, image, features_list = [], **kwargs):
        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image_features, patch_features_list = self.model.encode_image(
            image, 
            features_list=features_list or self.cfg.features_list,
        )
        return image_features, patch_features_list
