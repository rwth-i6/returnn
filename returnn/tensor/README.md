Here we provide the `Tensor` and `Dim` classes,
which are the basic building blocks in RETURNN for various backends:

- The low-level TensorFlow (graph-based) backend, so it wraps `tf.Tensor`.
- The low-level PyTorch (eager-based) backend, so it wraps `torch.Tensor`.
- The high-level RETURNN-TF (graph-based) backend,
  where it wraps the output of a layer
  (it's a string, corresponding to the layer name,
  as identified via the network dict).

See the `tensor` module docstring for more details.
