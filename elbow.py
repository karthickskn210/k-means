from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer


X, y = make_blobs(n_samples=3000, n_features=4, centers=3)


model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,9))

visualizer.fit(X)      
visualizer.poof() 
