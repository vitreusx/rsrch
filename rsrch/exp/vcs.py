from pathlib import Path

import rsrch
import wandb


class WandbVCS:
    def save(self):
        pkg_dir = Path(rsrch.__file__).parent
        wandb.run.log_code(str(pkg_dir))
