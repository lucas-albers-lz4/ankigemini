"""Test configuration and fixtures."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Add the parent directory to Python path to import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def tmp_html_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary HTML file for testing."""
    test_file = tmp_path / "test.html"
    test_file.write_text("""#separator:tab
#html:true
<div>
    <b>Question:</b><br>
    Test question
    <ul>
        <li>A. Option 1</li>
        <li class="correct">B. Option 2</li>
        <li>C. Option 3</li>
    </ul>
    <b>Explanation(s):</b>
    Test explanation
</div>""")
    yield test_file
