import asyncio
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
from Algoritmos.clustering_axo9 import OneSampleTTest  # Ajusta la ruta según tu proyecto
import numpy as np

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

        np.random.seed(42)
        data = np.random.normal(loc=50, scale=5, size=30)


        t_test: OneSampleTTest = OneSampleTTest(data=data, pop_mean=50, alpha=0.05, tail='two-tailed')

        _ = await t_test.persistify()

        results = t_test.run_distributed_test()


        print("\n[TEST One-sample t-test] Resultados:")
        for key, value in results.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
