# Import necessary libraries
import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Load and prepare the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train SOM
som_shape = (8, 8)  # 8x8 grid of neurons
som = MiniSom(x=som_shape[0], y=som_shape[1], input_len=X.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)

# Initialize weights
som.random_weights_init(X_scaled)

# Train the SOM
som.train_random(X_scaled, num_iteration=1000, verbose=True)


# Visualize the results
def plot_som_clusters():
    plt.figure(figsize=(10, 10))

    # Plot points
    for i, x in enumerate(X_scaled):
        w = som.winner(x)
        # Plot points with different colors for each class
        plt.plot(
            w[0] + 0.5 + 0.3 * np.random.rand() - 0.15,
            w[1] + 0.5 + 0.3 * np.random.rand() - 0.15,
            "o",
            markerfacecolor="None",
            markeredgecolor=["r", "g", "b"][y[i]],
            markersize=7,
        )

    plt.grid()
    plt.title("SOM Clustering of Iris Dataset")
    plt.show()


# Calculate and print quantization error
qe = som.quantization_error(X_scaled)
print(f"Quantization Error: {qe}")

# Plot the results
plot_som_clusters()

# Get cluster assignments for each data point
cluster_labels = [som.winner(x) for x in X_scaled]

# Create a dataframe with results
results_df = pd.DataFrame(
    {
        "sepal_length": X[:, 0],
        "sepal_width": X[:, 1],
        "petal_length": X[:, 2],
        "petal_width": X[:, 3],
        "true_species": y,
        "som_cluster": cluster_labels,
    }
)

print("\nFirst few rows of results:")
print(results_df.head())
