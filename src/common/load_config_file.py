import yaml
import os


class ConfigLoader:
    """
    Loads and caches a YAML configuration file.

    All paths stored in config.yaml are relative to the project root
    (the directory that contains config.yaml). Use ``resolve()`` to
    convert any config value to an absolute path at runtime.

    Usage::

        loader = ConfigLoader("config.yaml")
        cfg    = loader.config              # lazy-loaded, cached dict
        model  = loader.resolve(cfg["best_model"]["uc1"])
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._config: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict:
        """Return the config dict, loading it on first access."""
        if self._config is None:
            self._config = self.load()
        return self._config

    def load(self) -> dict:
        """Parse the YAML file and cache the result."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Config file not found: {self.file_path}")
        with open(self.file_path, "r") as fh:
            data = yaml.safe_load(fh)
        self._config = data
        return data

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    @property
    def base_dir(self) -> str:
        """Absolute path to the directory that contains config.yaml (= project root)."""
        return os.path.dirname(os.path.abspath(self.file_path))

    def resolve(self, path: str) -> str:
        """
        Convert a path from the config to an absolute path.

        If *path* is already absolute it is returned unchanged.
        Otherwise it is resolved relative to the directory that
        contains config.yaml (the project root).

        Args:
            path: A raw string value read from the config dict.

        Returns:
            Normalised absolute path string.

        Example::

            loader = ConfigLoader("config.yaml")
            abs_path = loader.resolve(loader.config["best_model"]["uc1"])
            # → "/home/user/project/train_results/uc1/.../best.pt"
        """
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))