"""
Python compatibility utils
"""

from __future__ import annotations


try:
    from typing import Protocol
except ImportError:
    try:
        from typing_extensions import Protocol
    except ImportError:

        class Protocol:
            """dummy"""

            pass
