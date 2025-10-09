from pathlib import Path
from urllib.request import urlretrieve
import pytest


BASE_URL = "https://raw.githubusercontent.com/AvantiShri/model_storage/d53ee8e/modisco/gkmexplain_scores"


@pytest.fixture(scope="session")
def data_ohe_hyps():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "ohe1.npz").exists():
        urlretrieve(f"{BASE_URL}/ohe1.npz", data_dir / "ohe1.npz")

    if not (data_dir / "hypscores1.npz").exists():
        urlretrieve(f"{BASE_URL}/hypscores1.npz", data_dir / "hypscores1.npz")

    return data_dir / "ohe1.npz", data_dir / "hypscores1.npz"

