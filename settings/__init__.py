"""
settings/

The Gelem settings mechanism: the values a researcher can tune per machine,
their defaults and bounds, and the persistence around them.

This package is Qt-free EXCEPT settings/qsettings_backend.py, which is the
one place PySide6 is imported. Nothing outside main.py and this package may
import from here (guarded by tests/test_settings.py).

docs/architecture.md section 9 is the single authority for what each value
means and which take effect immediately versus on restart. This package does
not restate that.
"""
