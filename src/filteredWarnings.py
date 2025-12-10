import warnings

def suppress_noisy_warnings() -> None:
    """Suppress specific noisy warnings from third-party libraries."""
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"faster_whisper.feature_extractor",
    )
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"pkg_resources"
    )
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"webrtcvad"
    )
