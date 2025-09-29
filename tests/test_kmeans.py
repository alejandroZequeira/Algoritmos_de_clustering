from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Algoritmos.clustering_axo2 import KMeansAnalyzer 
import asyncio

def dem():
    dem = DistributedEndpointManager()
    dem.add_endpoint(
        endpoint_id  = "axo-endpoint-0",
        hostname     = "localhost",
        protocol     = "tcp",
        req_res_port = 16667,
        pubsub_port  = 16666
    )
    return dem


async def test_create_kmeans():
    np.random.seed(42)

    data = pd.DataFrame({
        "X": np.random.rand(30) * 10,
        "Y": np.random.rand(30) * 10
    })

    with AxoContextManager.distributed(endpoint_manager=dem()) as dcm:
        kmeans_model: KMeansAnalyzer = KMeansAnalyzer(data=data)
        

        kmeans_model.show_columns()
        

        _ = await kmeans_model.persistify()

        result = kmeans_model.apply_kmeans(columnas=["X", "Y"], n_clusters=3)
        print(result)
        
        # Optional: show data
        # kmeans_model.show_data()

        # Optional: plot clusters
        # plt.figure(figsize=(8,6))
        # for cluster in range(3):
        #     cluster_data = data[result[1] == cluster]
        #     plt.scatter(cluster_data["X"], cluster_data["Y"], label=f"Cluster {cluster}")
        #
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.title("KMeans Clusters")
        # plt.legend()
        # plt.show()

if __name__=="__main__":
    asyncio.run(test_create_kmeans())
