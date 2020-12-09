.. _search:


######
Search
######

RETURNN can perform beam search on an arbitrary network architecture with an arbitrary number of outputs.
Mostly due to this fact, there is no single place in the code where beam search happens.
The following classes and functions are related to beam search:

    - :class:`returnn.tf.layers.rec.ChoiceLayer`

    - :class:`returnn.tf.layers.rec.DecideLayer`

    - :class:`returnn.tf.util.data.SearchBeam`

    - :class:`returnn.tf.layers.base.SearchChoices`

    - :class:`returnn.tf.layers.basic.SelectSearchSourcesLayer`

    - :func:`returnn.tf.layers.rec._SubnetworkRecCell._opt_search_resolve()`

    - :func:`returnn.tf.network.TFNetwork.get_search_choices()`

For an example implementation of search, please have a look at :ref:`recurrent_subnet_independent`.

Choice- and Decide Layer
------------------------

ChoiceLayer and DecideLayer are the only layers that actively manipulate the beam.
In the other layers the beam is hidden inside the batch dimension and the layers aren't even aware of this.

During search, ChoiceLayer creates a beam by taking the top (i.e. largest) k elements of the input vector
(which is typically the output of a softmax).
If the input has a beam itself then the output will contain the top k elements found in any of the vectors in the
incoming beam.
DecideLayer simply outputs the first best entry in the beam.
This is used after the actual search to output the first best hypothesis
(although, if you skip it, you can also output the n-best list).

During training (by default) there is no beam, ChoiceLayer outputs the ground truth label and DecideLayer does nothing.

Beam Selection
--------------

In addition to those two layers, there is logic between the layers that manipulates the beam:
ChoiceLayers can occur at any place in the network.
Therefore, in general, all layers operate on a different beam.
When a layer has several inputs, it must be ensured that the incoming beams correspond to each other.
This means, that all the incoming beams have the same size and all the n-th entries in each of the beams derive from a
common origin (= ChoiceLayer) somewhere along the network graph.
For this, the graph is parsed for each layer, the most recent ChoiceLayer is determined and all other inputs are
"translated" to this most recent beam.
In more detail, this means that we trace back the dependencies of each entry in the most recent beam until we get to
another ChoiceLayer.
Here we collect the single corresponding entry we arrived at.
Doing that for all entries in the most recent beam and for all ChoiceLayers in the network we create new "selected"
beams for the other inputs, which are then used instead of the original incoming ones as sources for the current layer.
In the code this happens in:

    | :func:`network._create_layer() <returnn.tf.network._create_layer()>` ->
    | :func:`network._create_layer_desc() <returnn.tf.network._create_layer_desc()>` ->
    | :func:`SearchChoices.translate_to_common_search_beam() <returnn.tf.layers.base.SearchChoices.translate_to_common_search_beam()>` ->
    | :func:`SearchChoices.translate_to_this_search_beam() <returnn.tf.layers.base.SearchChoices.translate_to_this_search_beam()>` ->
    | :func:`SelectSearchSourcesLayer.select_if_needed() <returnn.tf.layers.basic.SelectSearchSourcesLayer.select_if_needed()>` ->
    | :func:`SelectSearchSourcesLayer.__init__() <returnn.tf.layers.basic.SelectSearchSourcesLayer.__init__()>` ->
    | :func:`select_src_beams() <returnn.tf.util.basic.select_src_beams()>`

Backtracking
------------

Finally, when there is a beam inside a recurrent layer (this is actually the most common place where it occurs),
there is an additional step in which the beams of the outputs of the recurrent layer are resolved over time after all
time steps are evaluated.
This is exactly what is better known as backtracking, i.e. we create the full n-best sequences for all outputs,
instead of outputting the contents of the beam at each step in time.
This is implemented in:

    | :func:`_SubnetworkRecCell.get_output() <returnn.tf.layers.rec._SubnetworkRecCell.get_output()>` ->
    | :func:`_SubnetworkRecCell._construct_output_layers_moved_out() <returnn.tf.layers.rec._SubnetworkRecCell._construct_output_layers_moved_out()>` ->
    | :func:`_SubnetworkRecCell.get_loop_acc_layer() <returnn.tf.layers.rec._SubnetworkRecCell.get_loop_acc_layer()>` ->
    | :func:`_SubnetworkRecCell._opt_search_resolve() <returnn.tf.layers.rec._SubnetworkRecCell._opt_search_resolve()>`

For all this, the utility function that parses the dependency graph for the most recent ChoiceLayers is
:func:`returnn.tf.network.TFNetwork.get_search_choices()`.
