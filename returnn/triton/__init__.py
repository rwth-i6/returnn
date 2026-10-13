"""
Triton kernels shared by the backends.
Each module here holds only ``@triton.jit`` kernels and backend-neutral launch helpers,
the backends keep their launchers and gradients in ``returnn/<backend>/util``.
"""
