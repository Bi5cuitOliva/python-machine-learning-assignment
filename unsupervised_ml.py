
"""
Part B: Unsupervised Machine Learning
Customer Segmentation Using K-Means
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    # Step 1: Prepare customer data.
    # Each row has:
    # [Monthly Spending in KES, Monthly Visits]

    customer_data = [
        [1000, 2],
        [1500, 3],
        [2000, 2],
        [5000, 6],
        [6000, 7],
        [5500, 5],
        [12000, 12],
        [15000, 14],
        [13000, 11],
        [3000, 4]
    ]

    customer_ids = list(range(1, len(customer_data) + 1))

    # Step 2: Convert the dataset into a NumPy array.
    X = np.array(customer_data, dtype=float)

    # Step 3: Scale the features using StandardScaler.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 4: Create the K-Means clustering model.
    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    # Step 5: Train the model and assign clusters.
    cluster_labels = model.fit_predict(X_scaled)

    # Step 6: Display each customer's assigned cluster.
    print("Customer cluster assignments:")

    for customer_id, cluster in zip(customer_ids, cluster_labels):
        print(
            f"Customer {customer_id} belongs to Cluster {cluster}"
        )

    # Step 7: Display the total customers in each cluster.
    print("\nNumber of customers in each cluster:")

    for cluster_id in range(3):
        count = int(np.sum(cluster_labels == cluster_id))
        print(f"Cluster {cluster_id}: {count} customer(s)")

    # Bonus: Plot the customer groups using different colours.
    plt.figure(figsize=(8, 5))

    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=cluster_labels,
        cmap="viridis",
        s=100
    )

    # Label each point with its customer ID.
    for customer_id, (spending, visits) in zip(customer_ids, X):
        plt.annotate(
            str(customer_id),
            (spending, visits),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.title("Customer Segmentation Using K-Means")
    plt.xlabel("Monthly Spending (KES)")
    plt.ylabel("Monthly Visits")

    plt.legend(
        *scatter.legend_elements(),
        title="Cluster"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
