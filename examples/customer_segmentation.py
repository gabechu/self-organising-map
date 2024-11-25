import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load data
url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
df = pd.read_csv(url)
# Select features for clustering
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# Step 2: Initialize SOM
som_grid_size = (10, 10)  # 10x10 grid
som = MiniSom(
    x=som_grid_size[0], y=som_grid_size[1], input_len=len(features), sigma=1.0, learning_rate=0.5, random_seed=42
)

# Initialize weights
som.random_weights_init(X_scaled)

# Train SOM
n_iterations = 10000
som.train(X_scaled, n_iterations)


# Step 3: Visualize Results
def plot_som_results(som, X, feature_names):
    plt.figure(figsize=(12, 8))

    # Plot 1: Distance Map
    plt.subplot(1, 2, 1)
    plt.pcolor(som.distance_map().T, cmap="bone_r")
    plt.colorbar()
    plt.title("SOM Distance Map")

    # Plot 2: Customer Distribution
    plt.subplot(1, 2, 2)

    # Get winner coordinates for each customer
    winners = np.array([som.winner(x) for x in X])

    # Create scatter plot
    plt.scatter(
        winners[:, 0] + 0.5, winners[:, 1] + 0.5, c=X[:, 1], cmap="viridis", alpha=0.7  # Color by spending score
    )
    plt.colorbar(label="Spending Score")
    plt.title("Customer Distribution on SOM")
    plt.tight_layout()
    plt.show()


# Plot results
plot_som_results(som, X_scaled, features)


# Step 4: Identify Customer Segments
def get_customer_segments(som, X_scaled, X_original):
    # Calculate average values for each node
    node_averages = {}
    for i, (x_scaled, x_orig) in enumerate(zip(X_scaled, X_original)):
        bmu = tuple(som.winner(x_scaled))
        if bmu not in node_averages:
            node_averages[bmu] = {"count": 0, "income": 0, "spending": 0}
        node_averages[bmu]["count"] += 1
        node_averages[bmu]["income"] += x_orig[0]
        node_averages[bmu]["spending"] += x_orig[1]

    # Calculate averages
    for bmu in node_averages:
        count = node_averages[bmu]["count"]
        node_averages[bmu]["income"] /= count
        node_averages[bmu]["spending"] /= count

    return node_averages


segments = get_customer_segments(som, X_scaled, X)

# Print segment insights
for bmu, data in segments.items():
    if data["count"] > 5:  # Show only significant segments
        print(f"\nSegment at node {bmu}:")
        print(f"Number of customers: {data['count']}")
        print(f"Average income: ${data['income']:.2f}k")
        print(f"Average spending score: {data['spending']:.2f}")


# Step 5: Make Predictions for New Customers
def predict_segment(customer_data, som, scaler):
    # Scale new customer data
    scaled_data = scaler.transform([customer_data])

    # Find BMU
    bmu = som.winner(scaled_data[0])
    return bmu


# Example new customer
new_customer = [60, 75]  # Income: 60k, Spending Score: 75
segment = predict_segment(new_customer, som, scaler)
print(f"New customer belongs to segment at node {segment}")
