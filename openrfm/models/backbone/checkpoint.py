# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""Loading the released backbone file (`openrfm train backbone export`)."""
import torch

from .config import RTConfig
from .transformer import RelationalTransformer


def load_backbone(path, device, *, config: RTConfig = RTConfig(), attention: str = "dense") -> RelationalTransformer:
    """The backbone of a released backbone file, strictly loaded, in fp32 and eval mode on ``device``."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = RelationalTransformer(config, attention=attention)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device=device, dtype=torch.float32).eval()
