import asyncio
import numpy as np
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
from Algoritmos.clustering_axo6 import KMedoids   


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
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(loc=[2, 2], scale=0.5, size=(10, 2)),
        np.random.normal(loc=[7, 7], scale=0.5, size=(10, 2)),
        np.random.normal(loc=[2, 7], scale=0.5, size=(10, 2))
    ])

    with AxoContextManager.distributed(endpoint_manager=dem()) as dcm:
        kmedoids_model: KMedoids = KMedoids(n_clusters=3, max_iter=100, random_state=42)

        _ = await kmedoids_model.persistify()

        resultados = kmedoids_model.fit(X)

        print("[TEST] Resultados KMedoids:", resultados)
        print("[TEST] Medoids encontrados:", resultados["Medoids"])
        print("[TEST] Labels asignados:", resultados["Labels"])


if __name__ == "__main__":
    asyncio.run(main())
