import numpy as np
import matplotlib.pyplot as plt


class Som:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.weights = None
        self.grid_coords = self._get_grid_coords()

    def _get_grid_coords(self) -> np.ndarray:
        coords = np.indices((self.height, self.width))
        return coords.transpose(1, 2, 0)

    def _initialize_weights(self, n_features: int, seed: int = 42) -> np.ndarray:
        np.random.seed(seed)
        weights = np.random.random((self.width, self.height, n_features))
        return weights

    def _find_bmu(self, weights: np.ndarray, input_vector: np.ndarray) -> tuple:
        bmu = np.argmin(np.sum((weights - input_vector) ** 2, axis=2))
        bmu_x, bmu_y = np.unravel_index(bmu, (self.width, self.height))
        return bmu_x, bmu_y

    def _calculate_influence(self, bmu_x: int, bmu_y: int, σt: float) -> np.ndarray:
        # vectorized
        bmu_coords = np.array([bmu_x, bmu_y])
        dist_to_bmu = np.sqrt(np.sum((self.grid_coords - bmu_coords) ** 2, axis=2))

        θt = np.exp(-(dist_to_bmu**2) / (2 * (σt**2)))
        return θt

    def _update_weights(self, weights: np.ndarray, input_vector: np.ndarray, θt: np.ndarray, αt: float) -> np.ndarray:
        θt = θt[:, :, np.newaxis]
        input_vector = input_vector.reshape(1, 1, -1)
        new_weights = weights + αt * θt * (input_vector - weights)
        return new_weights

    def train(self, input_data, n_max_iterations) -> np.ndarray:
        # validation
        if len(input_data.shape) != 2:
            raise ValueError("Input data must be 2D")

        # initialization
        ε = 1e-5
        α0 = 0.1
        σ0 = max(self.width, self.height) / 2
        if σ0 == 1:
            σ0 += ε
        λ = n_max_iterations / np.log(σ0)
        n_features = input_data.shape[-1]

        weights = self._initialize_weights(n_features)

        for t in range(n_max_iterations):
            σt = σ0 * np.exp(-t / λ)
            αt = α0 * np.exp(-t / λ)
            for vt in input_data:
                bmu_x, bmu_y = self._find_bmu(weights, vt)
                θt = self._calculate_influence(bmu_x, bmu_y, σt)
                weights = self._update_weights(weights, vt, θt, αt)

        self.weights = weights

    def save(self, file_name: str):
        plt.imsave(file_name, self.weights)
