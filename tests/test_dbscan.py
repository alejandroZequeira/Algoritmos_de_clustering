import asyncio
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
from Algoritmos.clustering_axo3 import DBSCANClustering  

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
        dbscan_model: DBSCANClustering = DBSCANClustering(eps=1.5, min_samples=2)
        
        # Generamos datos de prueba
        dbscan_model.generate_data(n_samples=30, centers=3)
        
        # Persistimos
        _ = await dbscan_model.persistify()
        
        # Aplicamos DBSCAN
        resultados = dbscan_model.apply_dbscan()  # ya self.X tiene datos
        print(resultados)
        

if __name__ == "__main__":
    asyncio.run(main())
