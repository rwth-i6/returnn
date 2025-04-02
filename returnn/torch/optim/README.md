Here we can put some arbitrary external optimizers.
It might be copied from some existing code, or our own implementation.
It might also happen that some of these will be added to later versions of PyTorch.
So, regarding the user config, the optimizers here should be differentiated
by having the full module name, e.g. like ``returnn.torch.optim.lion.Lion``.
