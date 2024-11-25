# kohonen.py
import matplotlib.pyplot as plt
import numpy as np


def train(input_data, n_max_iterations, width, height):
    σ0 = max(width, height) / 2
    α0 = 0.1
    np.random.seed(42)
    weights = np.random.random((width, height, 3))
    λ = n_max_iterations / np.log(σ0)
    print("λ: ", λ)

    for t in range(n_max_iterations):
        σt = σ0 * np.exp(-t / λ)
        αt = α0 * np.exp(-t / λ)

        for vt in input_data:
            bmu = np.argmin(np.sum((weights - vt) ** 2, axis=2))
            bmu_x, bmu_y = np.unravel_index(bmu, (width, height))
            for x in range(width):
                for y in range(height):
                    di = np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))
                    θt = np.exp(-(di**2) / (2 * (σt**2)))
                    weights[x, y] += αt * θt * (vt - weights[x, y])
    return weights


def test_data_with_4_features():
    input_data = np.random.random((10, 4))
    train(input_data, 100, 10, 10)


def test_divide_zero():
    input_data = np.random.random((10, 3))
    result = train(input_data, 100, 2, 2)

    print(result)


if __name__ == "__main__":
    # test_data_with_4_features()
    # test_divide_zero()
    input_data = np.ones((3, 3))
    image_data = train(input_data, 100, 3, 3)

    plt.imsave("100.png", image_data)
