Most code here originates from `returnn_common.nn`.

Instead of
```python
from returnn_common import nn
```
now we have:
```python
import returnn.frontend as rf
import returnn.tf.frontend_layers as rfl
from returnn.tensor import Tensor, Dim

from . import _utils  # for RFL internal code
```
So the `nn` part is split into `rf` and `rfl`.
The `returnn_common.nn` basically was the basis for the API of the RETURNN frontend (RF).
The RF now supports multiple backends.
See here for some history and discussions:
-
- https://github.com/rwth-i6/returnn_common/issues/252
- https://github.com/rwth-i6/returnn/issues/1120
- https://github.com/rwth-i6/returnn/pull/1261
- https://github.com/rwth-i6/returnn/issues/1165

Some classes are renamed:

* `nn.NameCtx` -> RFL specific, `rfl.Layer`
* `nn.Tensor` -> `Tensor`
* `nn.Dim` -> `Dim`
* `nn.LayerState` -> `rf.State`
* `nn.make_layer` -> RFL specific, `rfl.make_layer`
* `nn.Module` -> `rf.Module`
* `nn.copy` -> only needed for RFL, `_utils.copy`
* `nn.constant_value` -> `_utils.constant_value`,
  but we should maybe introduce such function in RF
