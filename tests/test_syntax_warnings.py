# -*- coding: utf-8 -*-
"""
Every python file of LaueTools compiles without SyntaxWarning (shown since python 3.12 the first time a file is
compiled): invalid escape sequences such as "\\." in a normal string (use a raw string r"\\."), "is" / "is not" with
a literal (use == / !=), a missing comma between list items...
Invalid escape sequences are planned to become errors in a future python version.
"""
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "LaueTools").rglob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(ROOT)) for p in FILES])
def test_compiles_without_warning(path):
    source = path.read_text(encoding="utf-8", errors="replace")
    with warnings.catch_warnings():
        # SyntaxWarning (python >= 3.12) or DeprecationWarning (invalid escape sequences, python < 3.12)
        warnings.simplefilter("error", SyntaxWarning)
        warnings.simplefilter("error", DeprecationWarning)
        compile(source, str(path), "exec")
