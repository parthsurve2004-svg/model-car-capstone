"""Resolve local traffic-sign artifacts without fetching weights from a registry."""
import importlib.util
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[2] / 'models'


def ncnn_available():
    return importlib.util.find_spec('ncnn') is not None


def resolve_sign_model(explicit=None, *, disabled=False, bundle_root=None):
    """Return an existing artifact Path, or None if no bundle was installed.

    An explicit missing/unsupported artifact is an error, never an instruction
    to Ultralytics to download a model with a matching registry name.
    """
    if disabled:
        if explicit is not None:
            raise ValueError('Choose --disable-traffic-signs or --traffic-sign-model, not both')
        return None
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f'Traffic-sign model does not exist: {candidate}')
        if candidate.is_file() and candidate.suffix.lower() == '.pt':
            return candidate
        if candidate.is_dir() and all((candidate / name).is_file() for name in
                                      ('model.ncnn.bin', 'model.ncnn.param', 'metadata.yaml')):
            if not ncnn_available():
                raise RuntimeError('NCNN runtime unavailable; install ncnn or select local .pt weights')
            return candidate
        raise ValueError('Select an existing .pt checkpoint or complete NCNN export directory')
    root = BUNDLE_ROOT if bundle_root is None else Path(bundle_root)
    ncnn = root / 'traffic_signs_ncnn_model'
    if ncnn_available() and all((ncnn / name).is_file() for name in
                                ('model.ncnn.bin', 'model.ncnn.param', 'metadata.yaml')):
        return ncnn.resolve()
    checkpoint = root / 'traffic_signs.pt'
    return checkpoint.resolve() if checkpoint.is_file() else None
