import asyncio
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
from Algoritmos.clustering_axo8 import KolmogorovSmirnovTest 
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
        data = np.random.normal(loc=0, scale=1, size=100)
        ks_model: KolmogorovSmirnovTest = KolmogorovSmirnovTest(data=data, dist='norm', alpha=0.05)

        _ = await ks_model.persistify()

        results = ks_model.run_distributed_test()

        print("\n[TEST KS] Resultados del test de Kolmogorov-Smirnov:")
        for key, value in results.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
