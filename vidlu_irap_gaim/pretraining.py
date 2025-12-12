"""
Pretraining configuration helpers for IRAP GAIM models.
"""
from __future__ import annotations

from pathlib import Path
import os


def vistas_params_spec(vistas_weights_path: str | Path | None = None) -> str:
    """
    Returns a ViDLU parameter specification string for loading Vistas-pretrained weights.
    
    Maps `backbone.*` keys from vistas.pt to `frame_encoder.resnet.*` in the model.
    Use with `pretrained=True` in ResNetEncoder to load ImageNet first, then Vistas.
    
    **Note**: This is a helper for generating the string. The `--params` argument expects
    a literal string, not a function call. Use this function to print the spec, then
    copy it into your command, or construct the string directly.
    
    Args:
        vistas_weights_path: Path to vistas.pt. If None, uses `IRAP_HOME/weights/vistas.pt`.
    
    Returns:
        ViDLU parameter spec: `"id[backbone]->frame_encoder.resnet:/path/to/vistas.pt"`
    
    Example:
        >>> print(vistas_params_spec('/path/to/vistas.pt'))
        id[backbone]->frame_encoder.resnet:/path/to/vistas.pt
        
        # Then use in command:
        # --params "id[backbone]->frame_encoder.resnet:/path/to/vistas.pt"
    """
    if vistas_weights_path is None:
        irap_home = os.environ.get("IRAP_HOME")
        if not irap_home:
            raise FileNotFoundError(
                "IRAP_HOME not set. Provide vistas_weights_path or set IRAP_HOME "
                "and place vistas.pt at IRAP_HOME/weights/vistas.pt"
            )
        vistas_weights_path = Path(irap_home) / "weights" / "vistas.pt"
    
    vistas_path = Path(vistas_weights_path)
    if not vistas_path.exists():
        raise FileNotFoundError(f"vistas.pt not found: {vistas_path}")
    
    return f"id[backbone]->frame_encoder.resnet:{vistas_path.resolve()}"

