import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def generate_sample_data(apply_noise=False):
    """
    Generate sample data for clustering algorithms.
    """
    np.random.seed(0)
    cluster_1 = np.random.normal(loc=0.0, scale=0.5, size=(100, 2))
    cluster_2 = np.random.normal(loc=2.5, scale=0.5, size=(100, 2))
    cluster_3 = np.random.normal(loc=5.0, scale=0.5, size=(100, 2))
    if apply_noise:
        noise = np.random.uniform(low=-1, high=6, size=(50, 2))  # Generate uniform noise
        data = np.vstack((cluster_1, cluster_2, cluster_3, noise))
    else:
        data = np.vstack((cluster_1, cluster_2, cluster_3))
        
    return data

def apply_kmeans(data, n_clusters=3):
    """
    Apply K-Means clustering algorithm.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return labels, centers

def apply_dbscan(data, eps=0.3, min_samples=5):
    """
    Apply DBSCAN clustering algorithm.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    # Calculate cluster centers
    unique_labels = set(labels)
    centers = []
    for label in unique_labels:
        if label != -1:  # Ignore noise points
            cluster_points = data[labels == label]
            center = cluster_points.mean(axis=0)
            centers.append(center)
    centers = np.array(centers)  # Convert list of centers to NumPy array
    return labels, centers

def apply_gmm(data, n_components_range=range(1, 10), threshold=0.6):
    """
    Apply Gaussian Mixture Model (GMM) clustering algorithm.
    """
    bic_scores = []
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))
    
    # Find the optimal number of components based on the lowest BIC score
    optimal_n_components = n_components_range[np.argmin(bic_scores)]
    
    # Fit GMM with the optimal number of components
    gmm_optimal = GaussianMixture(n_components=optimal_n_components)
    gmm_optimal.fit(data)
    labels = gmm_optimal.predict(data)
    centers = gmm_optimal.means_
    probs = gmm_optimal.predict_proba(data)
    
    # Identify noise based on threshold
    noise = np.max(probs, axis=1) < threshold
    return labels, centers, noise, bic_scores, optimal_n_components

def plot_results(data, labels, centers, title, noise=None):
    """
    Plot clustering results.
    """
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    if noise is not None:
        plt.scatter(data[noise, 0], data[noise, 1], c='red', marker='x', label='Noise')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    if len(centers)>0:
        plt.scatter(centers[:, 0], centers[:, 1], s=300, c='blue', marker='X', label='Centers')
        plt.legend()
    plt.show()

def main():
    # Generate sample data
    data = generate_sample_data()
    
    # Apply K-Means
    kmeans_labels, kmeans_centers = apply_kmeans(data)
    plot_results(data, kmeans_labels, kmeans_centers, 'K-Means Clustering')
    
    # Apply DBSCAN
    dbscan_labels, dbscan_centers = apply_dbscan(data)
    plot_results(data, dbscan_labels, dbscan_centers, 'DBSCAN Clustering with Cluster Centers')
    
    # Apply GMM
    gmm_labels, gmm_centers, gmm_noise, bic_scores, optimal_n_components = apply_gmm(data)
    plot_results(data, gmm_labels, gmm_centers, 'GMM Clustering with Noise Identification', noise=gmm_noise)
    
    # Plot BIC scores
    plt.plot(range(1, 10), bic_scores, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score')
    plt.title('BIC Scores for Different n_components')
    plt.show()

if __name__ == "__main__":
    main()

