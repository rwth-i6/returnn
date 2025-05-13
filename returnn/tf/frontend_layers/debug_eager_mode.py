"""
Debug eager mode
"""

_debug_eager_mode_enabled = False


def enable_debug_eager_mode():
    """
    For debugging.

    Enables TF eager mode.
    Also, all layers will directly be created, and then due to TF eager mode directly evaluated.
    """
    global _debug_eager_mode_enabled
    import tensorflow as tf

    tf.compat.v1.enable_eager_execution()
    _debug_eager_mode_enabled = True


def disable_debug_eager_mode():
    """
    For debugging.

    Enables TF eager mode.
    Also, all layers will directly be created, and then due to TF eager mode directly evaluated.
    """
    global _debug_eager_mode_enabled
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    _debug_eager_mode_enabled = False


def is_debug_eager_mode_enabled() -> bool:
    """
    :return: True if debug eager mode is enabled.
    """
    return _debug_eager_mode_enabled
