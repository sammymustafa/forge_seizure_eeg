"""The interface every dataset adapter implements.

The point of this layer is that an analysis written against it runs unchanged on
CHB-MIT (scalp proxies for wearables), SeizeIT2 (real behind-the-ear wearable)
and eventually TUH — while the adapter stays responsible for telling the caller
*what it does not know*, through :meth:`DatasetAdapter.label_availability`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Recording, SeizureEvent


class DatasetAdapter(ABC):
    """Minimum surface an analysis needs from a dataset."""

    #: short identifier used in the ``dataset`` column of harmonised tables
    name: str = "unnamed"

    #: True when the EEG comes from real wearable hardware, False when the
    #: wearable configuration is only a scalp-position proxy. Results from the
    #: two must never be pooled or plotted as interchangeable measurements.
    is_wearable: bool = False

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ---------------------------------------------------------------- data
    @abstractmethod
    def recordings(self) -> list[Recording]:
        """Every recording the adapter can see on disk."""

    @abstractmethod
    def events(self, recording: Recording | None = None) -> list[SeizureEvent]:
        """Harmonised seizure events, for one recording or all of them."""

    @abstractmethod
    def load_signals(self, recording: Recording, channels: list[str]
                     ) -> tuple[np.ndarray, float, list[str]]:
        """Return ``(data[n_channels, n_samples], sfreq, channel_names)``.

        Implementations derive the requested channels where possible (e.g. a
        bipolar pair from a referential recording) and report which ones they
        actually produced.
        """

    # ---------------------------------------------------------------- meta
    @abstractmethod
    def montages(self) -> dict[str, list[str]]:
        """Electrode configurations this dataset supports, as derivations."""

    @abstractmethod
    def label_availability(self) -> dict[str, str]:
        """Which harmonised label fields this dataset actually carries.

        Values are :data:`seizure_eeg.schema.Provenance` strings. A field marked
        ``not_annotated`` must never be silently filled in by inference from the
        signals; doing so manufactures clinical labels.
        """

    # ---------------------------------------------------------------- utils
    def events_frame(self) -> pd.DataFrame:
        from ..schema import events_to_frame
        return events_to_frame(self.events())

    def recordings_frame(self) -> pd.DataFrame:
        from ..schema import recordings_to_frame
        return recordings_to_frame(self.recordings())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} root={self.root}>"
