import shutil
from contextlib import suppress
from pathlib import Path
from urllib.request import urlretrieve

import pytest

# the original test URLs are no longer reachable; the flickr URL still
# resolves, and docs/source/_images holds copies of the reference images
DOCS_IMAGES = Path(__file__).parent.parent / "docs" / "source" / "_images"
TEST_IMG_URL = "https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg"

# file name -> committed reference image (or the downloaded one, keyed None)
STAGED_IMAGES = {
    "test.jpg": "MonaLisa_Wikipedia.jpg",
    "test2.jpg": "MonaLisa_Wikipedia.jpg",
    "test_diff.jpg": "MonaLisa_Remix_Flickr.jpg",
    "test_url.jpg": None,
    "test1.jpg": None,
}


@pytest.fixture(scope="session")
def downloaded_image(tmp_path_factory):
    """Fetch the one still-reachable test image once per session.

    Tests that need it fail clearly if the network is unavailable.
    """
    dest = tmp_path_factory.mktemp("downloads") / "downloaded.jpg"
    with suppress(OSError):
        urlretrieve(TEST_IMG_URL, dest)
    return dest


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch, downloaded_image):
    """Run each test in an isolated cwd staged with the test images.

    Keeps downloads, rotated images, and other artifacts out of the repo.
    """
    monkeypatch.chdir(tmp_path)
    for name, source in STAGED_IMAGES.items():
        src = downloaded_image if source is None else DOCS_IMAGES / source
        if src.exists():
            shutil.copy(src, name)
    return tmp_path
