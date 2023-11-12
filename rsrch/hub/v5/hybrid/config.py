from dataclasses import dataclass

from . import cem, dreamer, impl, wm


@dataclass
class Config:
    wm: wm.Config
    dreamer: dreamer.Config
    cem: cem.Config
    impl: impl.Config
