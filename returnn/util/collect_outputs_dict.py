"""
Customized (derived) dict to pass as ``collected_outputs`` to some of the RF modules,
or potential other use cases.

You can predefine (by pattern) what kind of outputs you want to collect and store in this dict.
"""

from typing import Optional, Union, Sequence
import fnmatch


class CollectOutputsDict(dict):
    """
    Customized (derived) dict, where you can predefine (by key pattern)
    what kind of keys you want to collect and store in this dict.
    Other keys will be ignored.
    """

    def __init__(self, *args, allowed_key_patterns: Optional[Sequence[str]] = None, **kwargs):
        """
        Initialize the CollectOutputsDict.

        :param allowed_key_patterns:
            List of key patterns (with wildcards) that are allowed to be stored in the dict.
            If None, all keys are allowed.
        """
        super().__init__(*args, **kwargs)
        self.allowed_key_patterns = allowed_key_patterns

    def __setitem__(self, key, value):
        """
        Set an item in the dict if the key matches allowed patterns.
        """
        if self.is_key_allowed(key):
            super().__setitem__(key, value)

    def setdefault(self, key, default=None):
        """
        Set default value for a key if it matches allowed patterns.
        """
        if self.is_key_allowed(key):
            return super().setdefault(key, default)
        return None

    def update(self, mapping, **kwargs):
        """
        Update the dict with another mapping, only adding allowed keys.
        """
        assert not kwargs
        for key, value in mapping.items():
            if self.is_key_allowed(key):
                super().__setitem__(key, value)

    def is_key_allowed(self, key: str) -> bool:
        """
        Check if the key matches any of the allowed patterns.

        :param key:
        :return: True if the key is allowed, False otherwise.
        """
        if self.allowed_key_patterns is None:
            return True  # If no patterns defined, allow all keys
        for pattern in self.allowed_key_patterns:
            if fnmatch.fnmatch(key, pattern):
                return True
        return False


def is_key_allowed_in_collect_outputs_dict(collect_outputs: Union[CollectOutputsDict, dict], key: str) -> bool:
    """
    Check if a key is allowed in the given CollectOutputsDict.

    :param collect_outputs:
    :param key:
    :return: True if the key is allowed, False otherwise.
    """
    if isinstance(collect_outputs, CollectOutputsDict):
        return collect_outputs.is_key_allowed(key)
    return True  # If it's a regular dict, all keys are allowed
