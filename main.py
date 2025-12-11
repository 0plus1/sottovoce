from __future__ import annotations

from src.app import run_app
from src.config import get_settings
from src.filteredWarnings import suppress_noisy_warnings


def main() -> None:
    suppress_noisy_warnings()
    settings = get_settings()
    run_app(settings)


if __name__ == "__main__":
    main()
