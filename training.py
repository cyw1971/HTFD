"""Training entry that maps CLI args into the HTFD generation pipeline."""

from __future__ import annotations

from exp.exp_basic import Exp_Basic


class Exp_Generation(Exp_Basic):
    def train(self, setting: str = ""):
        self._set_env_from_args()
        # Import after env is set — exp_generation reads HTFD_* at import time.
        import exp.exp_generation as pipeline  # noqa: F401

        return None

    def test(self, setting: str = "", test: int = 0):
        return self.train(setting)


def run_training(args) -> None:
    exp = Exp_Generation(args)
    exp.train()
