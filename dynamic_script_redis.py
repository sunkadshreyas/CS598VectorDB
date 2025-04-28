import redis
import numpy as np
import time
import threading
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Dataset loading helpers
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_dataset(root_dir):
    xt = fvecs_read(f"{root_dir}/sift/sift_learn.fvecs")
    xb = fvecs_read(f"{root_dir}/sift/sift_base.fvecs")
    xq = fvecs_read(f"{root_dir}/sift/sift_query.fvecs")
    gt = ivecs_read(f"{root_dir}/sift/sift_groundtruth.ivecs")
    return xt, xb, xq, gt

# Connect to Redis
def connect_redis():
    return redis.Redis(host='localhost', port=6379)

# Create vector index
def create_redis_index(r, vector_dim):
    try:
        r.execute_command(
            'FT.CREATE', 'sift_idx',
            'ON', 'HASH',
            'PREFIX', '1', 'vec:',
            'SCHEMA',
            'vector', 'VECTOR', 'HNSW', '6',
            'TYPE', 'FLOAT32',
            'DIM', str(vector_dim),
            'DISTANCE_METRIC', 'L2'
        )
    except redis.exceptions.ResponseError as e:
        if "Index already exists" in str(e):
            print("Index already exists, skipping creation.")
        else:
            raise e

# Insert vectors
def insert_vectors(r, xb, ids):
    pipe = r.pipeline(transaction=False)
    for i, vec in zip(ids, xb):
        key = f"vec:{i}"
        if not r.exists(key):   # Check if key already exists
            pipe.hset(key, mapping={
                "vector": vec.tobytes()
            })
        # else: you can optionally log "skipped"
    pipe.execute()

# Delete vectors
def delete_vectors(r, ids):
    pipe = r.pipeline()
    for i in ids:
        pipe.delete(f"vec:{i}")
    pipe.execute()

# Search vectors
def search_vectors(r, xq, topk):
    results = []
    for query_vec in xq:
        res = r.execute_command(
            'FT.SEARCH', 'sift_idx',
            '*=>[KNN {} @vector $vec AS score]'.format(topk),
            'SORTBY', 'score',
            'RETURN', '0',
            'DIALECT', '2',
            'PARAMS', '2', 'vec', query_vec.tobytes()
        )
        ids = []
        for j in range(2, len(res), 2):
            id_str = res[j - 1]
            vec_id = int(id_str.decode('utf-8').split(':')[1])
            ids.append(vec_id)
        results.append(ids)
    return results

# Metric calculation
def compute_recall(results, ground_truth, k):
    correct = 0
    for res, gt in zip(results, ground_truth):
        correct += len(set(res[:k]).intersection(set(gt[:k])))
    return correct / (len(results) * k)

# Background search thread
def background_search_loop(r, xq, gt, topk, log, stop_event):
    while not stop_event.is_set():
        start = time.time()
        I = search_vectors(r, xq, topk)
        end = time.time()
        qps = xq.shape[0] / (end - start)
        latency = (end - start) * 1000
        recall = compute_recall(I, gt, topk)
        log['qps'].append(qps)
        log['latency'].append(latency)
        log['recall'].append(recall)
        time.sleep(0.5)

# Main evaluation
def simulate_dynamic_updates_redis(root_dir, pdf_path, update_percents=[25, 75], topk=10):
    xt, xb, xq, gt = load_dataset(root_dir)
    r = connect_redis()
    create_redis_index(r, xb.shape[1])

    base_size = xb.shape[0]
    insert_vectors(r, xb, np.arange(base_size))

    num_keys = sum(1 for _ in r.scan_iter(match="vec:*"))
    print(f"🔍 Found {num_keys} vector keys in Redis (expected {base_size})")
    assert num_keys == base_size, "Mismatch in inserted vector count!"

    start = time.time()
    I = search_vectors(r, xq, topk)
    end = time.time()

    elapsed_time = end - start
    baseline_qps = len(xq) / elapsed_time
    baseline_recall = compute_recall(I, gt, topk)

    print(f"\nBaseline - QPS: {baseline_qps:.2f}, Recall: {baseline_recall:.4f}")

    results_summary = {
        'update_percent': [0],
        'final_qps': [baseline_qps],
        'final_latency': [None],
        'final_recall': [baseline_recall]
    }

    pdf = PdfPages(pdf_path)

    for update_percent in update_percents:
        print(f"\nRunning with {update_percent}% updates...")
        num_updates = int(base_size * update_percent / 100)

        log = {'qps': [], 'latency': [], 'recall': []}
        stop_event = threading.Event()
        search_thread = threading.Thread(target=background_search_loop, args=(r, xq, gt, topk, log, stop_event))
        search_thread.start()

        time.sleep(2)

        # Delete a portion
        delete_vectors(r, np.arange(base_size - num_updates, base_size))
        print(f"Deleted {num_updates} vectors.")

        # Re-insert
        insert_vectors(r, xb[base_size - num_updates:], np.arange(base_size - num_updates, base_size))
        print(f"Re-inserted {num_updates} vectors.")

        time.sleep(5)
        stop_event.set()
        search_thread.join()

        results_summary['update_percent'].append(update_percent)
        results_summary['final_qps'].append(np.mean(log['qps'][-5:]))
        results_summary['final_latency'].append(np.mean(log['latency'][-5:]))
        results_summary['final_recall'].append(np.mean(log['recall'][-5:]))

        plt.figure(figsize=(10, 4))
        plt.plot(log['qps'], label='QPS')
        plt.plot(log['latency'], label='Latency (ms)')
        plt.xlabel("Time Interval")
        plt.ylabel("Value")
        plt.title(f"{update_percent}% Update - QPS & Latency (Redis)")
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(results_summary['update_percent'], results_summary['final_qps'], marker='o', label='QPS')
    plt.plot(results_summary['update_percent'], results_summary['final_latency'], marker='s', label='Latency (ms)')
    plt.plot(results_summary['update_percent'], results_summary['final_recall'], marker='^', label='Recall')
    plt.xlabel("Update Percent")
    plt.ylabel("Value")
    plt.title("QPS, Latency, Recall vs. Update Percent (Redis)")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()

    pdf.close()

# --- Main ---
if __name__ == "__main__":
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_dir / "dynamic_updates_redis.pdf"
    simulate_dynamic_updates_redis(".", pdf_path)
