

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Dict
import torch
import random, json, os
import numpy as np

@dataclass
class ModelConfig:
    # Core attributes
    model_name: str
    model_path: str
    device: str = "cuda"
    device_map: str = "auto"  # "auto" or None
    dtype: str = "float32"
    seed: int = 42
    max_new_tokens: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    batch_size: int = 1
    max_length: int = 512
    load_in_4bit: bool = False

    # --- dynamic attributes depending on model ---
    
    extra_args: Dict[str, Any] = field(default_factory=dict)


    # Factory constructor: split known vs unknown keys
    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ModelConfig":
        core_names = {f.name for f in fields(cls)}
        known = {k: v for k, v in cfg.items() if k in core_names}
        unknown = {k: v for k, v in cfg.items() if k not in core_names}
        known.setdefault("extra_args", {}).update(unknown)
        return cls(**known)

    # Attribute access only for extra_args (read)
    def __getattr__(self, name: str) -> Any:
        ea = super().__getattribute__("extra_args")  # avoid recursion
        if name in ea:
            return ea[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute '{name}'")

    # Attribute set only if it’s an extra arg; core attrs use normal setattr
    def __setattr__(self, name: str, value: Any) -> None:
        core_names = {f.name for f in fields(type(self))}
        if name in core_names or name == "extra_args":
            return super().__setattr__(name, value)
        # route unknown attributes into extra_args
        ea = super().__getattribute__("extra_args") if "extra_args" in self.__dict__ else None
        if ea is None:
            super().__setattr__("extra_args", {name: value})
        else:
            ea[name] = value

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.device_map = config.device_map if torch.cuda.is_available() else None
        self.dtype = config.dtype
        self._set_seed(config.seed)
        self.model_path = config.model_path
        self.model_name = config.model_name
        self.model = None
        self.processor = None

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        if dtype_str == "float32": return torch.float32
        if dtype_str in ("float16", "fp16"): return torch.float16
        if dtype_str in ("bfloat16", "bf16"): return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    def _set_seed(self, seed: int) -> None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    def log_model_info(self) -> None:
        print(f"[Model] {self.config.model_id}")
        print(f"[Device] {self.device}  [DType] {self.dtype}  [Seed] {self.config.seed}")
        # show extras deterministically
        if self.config.extra_args:
            print("[Extra Args]", json.dumps(self.config.extra_args, sort_keys=True))

    def predict(self, img_path: str, prompt: str):
        raise NotImplementedError("Subclasses must implement predict().")

    def __call__(self, video_path: str, prompt: str):
        return self.predict(video_path, prompt)