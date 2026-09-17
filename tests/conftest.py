import shutil

import pytest

from .helpers import BACKENDS, DOCS_IMAGES, TEST_IMG_URL, download, make_backend

# file name -> committed reference image (None = the downloaded one)
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
    download(TEST_IMG_URL, dest)
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


@pytest.fixture(params=BACKENDS)
def backend(request):
    """A search backend (client + driver), parametrized over ES and OpenSearch."""
    return make_backend(request.param)
