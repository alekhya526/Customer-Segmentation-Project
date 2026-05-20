
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('customer_segmentation.csv')

# Select features
X = df[['Annual Income', 'Spending Score']]

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Plot clusters
plt.figure(figsize=(6,5))
for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income'], cluster_data['Spending Score'], label=f'Cluster {cluster}')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using KMeans')
plt.legend()

plt.savefig('customer_clusters.png')
plt.show()
