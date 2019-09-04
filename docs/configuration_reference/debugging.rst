.. _configuration_debugging:

=========
Debugging
=========

debug_add_check_numerics_on_output
    If set to ``True``, should assert for ``inf`` and ``nan``.

debug_grad_summaries
    If set to ``True``, adds additional information about the gradients to the TensorBoard.

debug_print_layer_output_template
    If set to ``True``, print the layer template information during network construction.

debug_print_layer_output_shape
    If set to ``True``, print the layer shape information while the graph is executed.

debug_objective_loss_summaries
    If set to ``True``, adds the objective loss values (normalization only when activated, including loss scaling)
    to the TensorBoard.

debug_unnormalized_loss_summaries
    If set to ``True``, adds the unnormalized loss values to the TensorBoard

Also see :ref:`debugging`.
