"""Checkout-time import shim for the src-layout dyrca_rca package."""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "dyrca_rca"
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))
