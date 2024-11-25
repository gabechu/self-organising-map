from som_gchu import Som
import numpy as np


def test_get_grid_coords():
    som = Som(width=2, height=2)

    expected = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
    actual = som.grid_coords
    np.testing.assert_array_equal(actual, expected)


def test_find_bmu():
    som = Som(width=3, height=3)
    weights = np.ones((3, 3, 3))
    weights[1, 1] = [0, 0, 0]  # BMU

    input_vector = np.array([0, 0, 0])
    bmu_x, bmu_y = som._find_bmu(weights, input_vector)

    assert bmu_x == 1
    assert bmu_y == 1


def test_calculate_influence():
    som = Som(width=3, height=3)
    actual = som._calculate_influence(1, 1, σt=1.0)

    assert actual.shape == (3, 3)
    assert actual[1, 1] == 1.0  # Maximum influence at BMU
    assert np.all(actual <= 1.0)  # All values should be <= 1
    assert np.all(actual >= 0.0)  # All values should be >= 0


def test_update_weights():
    som = Som(3, 3)
    weights = np.ones((3, 3, 3))
    input_vector = np.zeros(3)
    θt = np.ones((3, 3))
    αt = 0.5

    new_weights = som._update_weights(weights, input_vector, θt, αt)

    # Check if weights moved in correct direction
    diff_old = np.abs(weights - input_vector)
    diff_new = np.abs(new_weights - input_vector)
    assert np.all(diff_new < diff_old)


def test_training_with_7_features():
    n_samples = 10
    n_features = 7
    som = Som(width=5, height=5)
    input_data = np.random.random((n_samples, n_features))

    som.train(input_data, n_max_iterations=100)

    assert som.weights.shape == (5, 5, n_features)
