from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
import pandas as pd
import asyncio
from Algoritmos.clustering_axo4 import ChiSquareTestAuto

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

async def test_chi_square():
    data = pd.DataFrame({
        "Pepsi": [30, 10],
        "Coca": [25, 15],
        "Fanta": [20, 20]
    })

    with AxoContextManager.distributed(endpoint_manager=dem()) as dcm:
        chi_test: ChiSquareTestAuto = ChiSquareTestAuto(data=data)

        _ = await chi_test.persistify()

        resultado = chi_test.run_test()

        print("\n=== RESULTADOS CHI-SQUARE ===")
        print(resultado)

if __name__ == "__main__":
    asyncio.run(test_chi_square())
