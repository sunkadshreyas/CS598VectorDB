import weaviate
import numpy as np
from threading import Thread, Event, Lock
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# 1. Connect to Weaviate
client = 

# 2. Define schema (run once)
class_obj = {
    "class": "SIFT",
    "vectorizer": "none",
    "properties": [
        {"name": "vector_id", "dataType": ["int"], "indexInverted": False},
    ]
}
if not client.schema.exists("SIFT"):
    client.schema.create_class(class_obj)

# Helpers to read the SIFT files (same as yours)
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

# 3. Bulk-import xb into Weaviate
def import_base(xb):
    with client.batch.configure(batch_size=256, dynamic=True) as batch:
        for idx, vec in enumerate(xb):
            obj = {"vector_id": idx}
            batch.add_data_object(obj, "SIFT", vector=vec.tolist())
    # wait for any background indexing to finish (if needed)
    client.batch.flush()

# 4. Search helper using GraphQL kNN
def weaviate_search(xq_batch, topk):
    """
    Sends one multi-vector search:
    """
    # Weaviate only supports single-vector search per query, so you might
    # want to parallelize or loop. Here’s a simple loop:
    results = []
    for vec in xq_batch:
        resp = client.query \
            .get("SIFT", ["vector_id"]) \
            .with_near_vector({"vector": vec.tolist(), "certainty": None}) \
            .with_limit(topk) \
            .do()
        ids = [int(r['vector_id']) for r in resp['data']['Get']['SIFT']]
        results.append(ids)
    return results

def compute_recall(results, gt, k):
    correct = 0
    for res, true in zip(results, gt):
        correct += len(set(res[:k]).intersection(set(true[:k])))
    return correct / (len(results) * k)

# 5. Background loop to hit the index continuously
def background_search_loop(xq, gt, topk, log, stop_event, lock):
    while not stop_event.is_set():
        start = time.time()
        with lock:
            res = weaviate_search(xq, topk)
        end = time.time()
        qps = len(xq) / (end - start)
        latency_ms = (end - start) * 1000
        recall = compute_recall(res, gt, topk)
        log['qps'].append(qps)
        log['latency'].append(latency_ms)
        log['recall'].append(recall)
        time.sleep(0.5)

# 6. Simulation of dynamic updates
def simulate_dynamic_updates_weaviate(root_dir, pdf_path, update_percents=[25, 75], topk=10):
    xt, xb, xq, gt = load_dataset(root_dir)
    # initialize Weaviate with full base
    import_base(xb)
    pdf = PdfPages(pdf_path)
    base_size = xb.shape[0]

    # measure baseline
    print("Measuring baseline...")
    start = time.time()
    baseline_res = weaviate_search(xq, topk)
    baseline_time = time.time() - start
    baseline_qps = len(xq) / baseline_time
    baseline_latency = baseline_time * 1000
    baseline_recall = compute_recall(baseline_res, gt, topk)
    print(f"Baseline QPS={baseline_qps:.1f}, Latency={baseline_latency:.1f}ms, Recall={baseline_recall:.4f}")

    summary = {
        'update_percent': [0],
        'qps': [baseline_qps],
        'latency': [baseline_latency],
        'recall': [baseline_recall],
    }

    for pct in update_percents:
        num = int(base_size * pct / 100)
        print(f"\nApplying {pct}% updates ({num} vectors)…")
        log = {'qps': [], 'latency': [], 'recall': []}
        lock = Lock()
        stop = Event()
        t = Thread(target=background_search_loop, args=(xq, gt, topk, log, stop, lock))
        t.start()

        # let searches warm up
        time.sleep(2)

        # 6a. Delete the last slice
        with lock:
            t0 = time.time()
            for vid in range(base_size - num, base_size):
                client.data_object.delete(str(vid), "SIFT")
            delete_lat = time.time() - t0
        print(f"Delete latency: {delete_lat:.3f}s")

        # 6b. Re-insert them
        with lock:
            t0 = time.time()
            with client.batch.configure(batch_size=256, dynamic=True) as b:
                for vid in range(base_size - num, base_size):
                    obj = {"vector_id": vid}
                    b.add_data_object(obj, "SIFT", vector=xb[vid].tolist())
            client.batch.flush()
            ins_lat = time.time() - t0
        print(f"Re-insert throughput: {num/ins_lat:.1f} vecs/sec")

        # let it settle & measure
        time.sleep(5)
        stop.set()
        t.join()

        summary['update_percent'].append(pct)
        summary['qps'].append(np.mean(log['qps'][-5:]))
        summary['latency'].append(np.mean(log['latency'][-5:]))
        summary['recall'].append(np.mean(log['recall'][-5:]))

        # plot this run
        plt.figure(figsize=(10,4))
        plt.plot(log['qps'], label='QPS')
        plt.plot(log['latency'], label='Latency (ms)')
        plt.title(f"{pct}% Updates")
        plt.xlabel("Interval")
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.close()

    # final summary plot
    plt.figure(figsize=(10,5))
    plt.plot(summary['update_percent'], summary['qps'], marker='o', label='QPS')
    plt.plot(summary['update_percent'], summary['latency'], marker='s', label='Latency (ms)')
    plt.plot(summary['update_percent'], summary['recall'], marker='^', label='Recall')
    plt.xlabel("Update %")
    plt.ylabel("Metric")
    plt.title("Weaviate Dynamic Updates Summary")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()
    pdf.close()

if __name__ == "__main__":
    out = Path("plots")
    out.mkdir(exist_ok=True)
    simulate_dynamic_updates_weaviate(".", out/"weaviate_dynamic.pdf")
