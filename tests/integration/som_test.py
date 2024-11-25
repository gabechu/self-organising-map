from som_gchu import Som
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_train_result():
    fake_data = np.ones((3, 3))
    som = Som(3, 3)
    som.train(fake_data, 100)
    actual = som.weights
    expected = np.array(
        [
            [[1.0, 1.0, 1.0], [1.0, 0.99999999, 0.99999999], [0.99910117, 0.9998723, 0.99961936]],
            [[1.0, 0.99999999, 1.0], [0.99999968, 0.99999852, 0.99999846], [0.99502439, 0.99576067, 0.99710429]],
            [
                [0.99945793, 0.99932365, 0.99962961],
                [0.99475684, 0.99568695, 0.99613917],
                [0.93180368, 0.973066, 0.89965749],
            ],
        ]
    )

    assert_array_almost_equal(actual, expected)
