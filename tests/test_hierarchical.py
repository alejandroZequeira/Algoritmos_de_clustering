import asyncio
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
import numpy as np
from Algoritmos.clustering_axo5 import HierarchicalClustering  


def dem():
    dem = DistributedEndpointManager()
    dem.add_endpoint(
        endpoint_id="axo-endpoint-0",
        hostname="localhost",
        protocol="tcp",
        req_res_port=16667,
        pubsub_port=16666
    )
    return dem

    

async def main():
    with AxoContextManager.distributed(endpoint_manager=dem()) as dcm:
         hierar_model:HierarchicalClustering = HierarchicalClustering()
        # Instancia del modelo
         hierar_model = HierarchicalClustering(method="ward")
         data = np.array([
            [1, 2], [1, 4], [1, 0],
            [4, 2], [4, 4], [4, 0],
            [10, 10], [10, 11], [11, 10],
            [20, 20]
        ])
    hierar_model.set_data(data)

            # Persistimos el objeto en Axo
    _ = await hierar_model.persistify()

        # Ejecutamos clustering jerárquico
    clusters = hierar_model.run(n_clusters=3)
    print(f"[TEST] Clusters encontrados: {clusters}")


if __name__ == "__main__":
    asyncio.run(main())
