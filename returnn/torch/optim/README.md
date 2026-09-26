Here we can put some arbitrary external optimizers.
It might be copied from some existing code, or our own implementation.
It might also happen that some of these will be added to later versions of PyTorch.
The optimizers here can be referenced in the user config
by short name (e.g. ``lion``, ``amuse``, ``multi``; ``torch.optim`` names take precedence)
or by full module name, e.g. ``returnn.torch.optim.lion.Lion``.
