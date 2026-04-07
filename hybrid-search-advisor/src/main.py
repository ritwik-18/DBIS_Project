from db.connection import get_db_connection
from src.search.executor import HybridSearchProxy

def run_system_tests():
    print("Initializing Workload-Aware Hybrid Search Proxy...\n")
    conn = get_db_connection()
    proxy = HybridSearchProxy(conn)
    
    dummy_vector = [0.5, 0.5, 0.5]

    print("--- TEST 1: Broad Filter (<30% Dropped) ---")
    proxy.execute_hybrid_search(dummy_vector, {'category': 'Computer Science'})

    print("\n--- TEST 2: THE DANGER ZONE (~50% Dropped) ---")
    proxy.execute_hybrid_search(dummy_vector, {'id >': 500})

    print("\n--- TEST 3: Strict Filter (>80% Dropped) ---")
    proxy.execute_hybrid_search(dummy_vector, {'author': 'Srinivasa Ramanujan'})

    conn.close()

if __name__ == "__main__":
    run_system_tests()