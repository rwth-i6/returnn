"""
main demo compile
"""

import tensorflow as tf
from returnn.tf.compat import v1 as tf_v1
from . import rna_loss, is_checked_out


def main():
    """main warp-rna demo"""
    print("Hello, running WarpRna demo")
    test_warprna_forward()


def test_warprna_forward():
    """test warp-rna forward"""
    assert is_checked_out()
    import numpy as np

    # computed using slow TF implementation (cross-check against numpy reference impl)
    expected_costs = np.array([2.6347387, 2.4651031])
    expected_grads = np.array(
        [
            [
                [[-0.34075904, -0.65924096, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[-0.09434381, -0.24641524, 0.0], [-0.4480959, 0.0, -0.2111451], [0.0, 0.0, 0.0]],
                [[0.0, -0.09434381, 0.0], [-0.25838017, 0.0, -0.43613094], [-0.2111451, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, -0.35272402], [-0.64727604, 0.0, 0.0]],
            ],
            [
                [[-0.6283351, -0.37166485, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[-0.26558593, -0.36274916, 0.0], [-0.23790276, -0.13376209, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.26558593, 0.0], [-0.26772842, -0.3329236, 0.0], [-0.13376209, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -0.53331435, 0.0], [-0.46668565, 0.0, 0.0]],
            ],
        ]
    )

    n_batch, n_time, n_target, n_vocab = 2, 4, 3, 3
    acts = np.array(
        [
            0.065357,
            0.787530,
            0.081592,
            0.529716,
            0.750675,
            0.754135,
            0.609764,
            0.868140,
            0.622532,
            0.668522,
            0.858039,
            0.164539,
            0.989780,
            0.944298,
            0.603168,
            0.946783,
            0.666203,
            0.286882,
            0.094184,
            0.366674,
            0.736168,
            0.166680,
            0.714154,
            0.399400,
            0.535982,
            0.291821,
            0.612642,
            0.324241,
            0.800764,
            0.524106,
            0.779195,
            0.183314,
            0.113745,
            0.240222,
            0.339470,
            0.134160,
            0.505562,
            0.051597,
            0.640290,
            0.430733,
            0.829473,
            0.177467,
            0.320700,
            0.042883,
            0.302803,
            0.675178,
            0.569537,
            0.558474,
            0.083132,
            0.060165,
            0.107958,
            0.748615,
            0.943918,
            0.486356,
            0.418199,
            0.652408,
            0.024243,
            0.134582,
            0.366342,
            0.295830,
            0.923670,
            0.689929,
            0.741898,
            0.250005,
            0.603430,
            0.987289,
            0.592606,
            0.884672,
            0.543450,
            0.660770,
            0.377128,
            0.358021,
        ],
        dtype=np.float32,
    )
    acts = np.reshape(acts, (n_batch, n_time, n_target, n_vocab))

    labels = np.array([[1, 2], [1, 1]], dtype=np.int32)
    input_lengths = np.array([4, 4], dtype=np.int32)
    label_lengths = np.array([2, 2], dtype=np.int32)

    acts_t = tf.convert_to_tensor(acts)
    labels_t = tf.convert_to_tensor(labels)
    input_lengths_t = tf.convert_to_tensor(input_lengths)
    label_lengths_t = tf.convert_to_tensor(label_lengths)
    log_probs = tf.nn.log_softmax(acts_t)
    costs_cuda = rna_loss(log_probs, labels_t, input_lengths_t, label_lengths_t)
    grads_cuda = tf.gradients(costs_cuda, [log_probs])[0]
    with tf_v1.Session() as session:
        (out_costs_cuda, out_grads_cuda) = session.run([costs_cuda, grads_cuda])

    print("[CUDA] costs:", out_costs_cuda)
    np.testing.assert_allclose(out_grads_cuda, expected_grads, rtol=1e-6)
    np.testing.assert_allclose(out_costs_cuda, expected_costs, rtol=1e-6)


if __name__ == "__main__":
    main()
