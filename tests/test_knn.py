import asyncio
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
from Algoritmos.clustering_axo7 import KNNAnalyzer 

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
        knn_model: KNNAnalyzer = KNNAnalyzer()


        _ = await knn_model.persistify()


        modelo, reporte = knn_model.run_test(
            n_samples=50,   # número de filas
            n_features=4,   # número de columnas de características
            n_classes=2,    # número de clases
            k=3             # número de vecinos
        )

        print("\n[TEST KNN] Reporte de clasificación:")
        for clase, metrics in reporte.items():
            print(f"{clase}: {metrics}")

if __name__ == "__main__":
    asyncio.run(main())
