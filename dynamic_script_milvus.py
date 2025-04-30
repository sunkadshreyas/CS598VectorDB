import numpy as np
import time
import threading
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

# ——— Helpers for loading SIFT
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_sift(root):
    xt = fvecs_read(f"{root}/sift/sift_learn.fvecs")
    xb = fvecs_read(f"{root}/sift/sift_base.fvecs")
    xq = fvecs_read(f"{root}/sift/sift_query.fvecs")
    gt = ivecs_read(f"{root}/sift/sift_groundtruth.ivecs")
    return xt, xb, xq, gt

def compute_recall(results, ground_truth, k):
    correct = 0
    for res, gt in zip(results, ground_truth):
        correct += len(set(res[:k]).intersection(gt[:k]))
    return correct / (len(results) * k)

# ——— Background search thread
def background_search_loop(collection, xq, gt, topk, log, stop_event, lock):
    while not stop_event.is_set():
        t0 = time.time()
        with lock:
            # perform search
            res = collection.search(
                xq.tolist(),
                "vector",
                params={"metric_type":"L2", "params":{"ef": 64}},
                limit=topk
            )
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000
        qps = len(xq) / (t1 - t0)
        # extract IDs
        ids = np.vstack([[r.id for r in hits] for hits in res])
        recall = compute_recall(ids, gt, topk)
        log['qps'].append(qps)
        log['latency'].append(latency_ms)
        log['recall'].append(recall)
        time.sleep(0.5)

# ——— Main driver
def simulate_milvus_dynamic(root_dir, pdf_path, update_percents=[25,75], topk=10):
    # 1. load data
    xt, xb, xq, gt = load_sift(root_dir)
    base_n, dim = xb.shape

    # 2. connect & setup
    connections.connect("default", host="127.0.0.1", port="19530")
    if utility.has_collection("sift_dynamic"):
        utility.drop_collection("sift_dynamic")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, "SIFT dynamic updates")
    coll = Collection("sift_dynamic", schema)

    # create HNSW index
    coll.create_index(
        "vector",
        {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 32, "efConstruction": 40}
        }
    )
    coll.load()

    # initial insert
    ids = np.arange(base_n, dtype=np.int64)
    coll.insert([ids, xb.tolist()])
    coll.flush()

    # baseline search
    t0 = time.time()
    res = coll.search(
        xq.tolist(),
        "vector",
        params={"metric_type":"L2", "params":{"ef":64}},
        limit=topk
    )
    t1 = time.time()
    baseline_qps = len(xq)/(t1-t0)
    baseline_lat = (t1-t0)*1000
    ids0 = np.vstack([[r.id for r in hits] for hits in res])
    baseline_rec = compute_recall(ids0, gt, topk)
    print(f"Baseline — QPS={baseline_qps:.2f}, Lat={baseline_lat:.2f}ms, Rec={baseline_rec:.4f}")

    summary = {
        'pct': [0],
        'qps': [baseline_qps],
        'lat': [baseline_lat],
        'rec': [baseline_rec]
    }
    pdf = PdfPages(pdf_path)

    # 3. dynamic updates
    for pct in update_percents:
        print(f"\n--- {pct}% update ---")
        n_upd = int(base_n * pct/100)

        log = {'qps':[], 'latency':[], 'recall':[]}
        lock = threading.Lock()
        stop = threading.Event()
        th = threading.Thread(
            target=background_search_loop,
            args=(coll, xq, gt, topk, log, stop, lock)
        )
        th.start()

        # give it a moment to stabilize
        time.sleep(2)

        # DELETE oldest n_upd IDs
        del_ids = ids[:n_upd]
        tdel = time.time()
        with lock:
            coll.delete(f"id in {del_ids.tolist()}")
            coll.flush()
        del_lat = time.time() - tdel
        print(f"Delete latency: {del_lat:.3f}s")

        # INSERT them back
        tins = time.time()
        with lock:
            coll.insert([del_ids, xb[:n_upd].tolist()])
            coll.flush()
        ins_t = time.time() - tins
        print(f"Insert throughput: {n_upd/ins_t:.1f} vecs/sec")

        # let it run for 5s more
        time.sleep(5)
        stop.set()
        th.join()

        summary['pct'].append(pct)
        summary['qps'].append(np.mean(log['qps'][-5:]))
        summary['lat'].append(np.mean(log['latency'][-5:]))
        summary['rec'].append(np.mean(log['recall'][-5:]))

        # plot this run
        plt.figure(figsize=(10,4))
        plt.plot(log['qps'], label='QPS')
        plt.plot(log['latency'], label='Latency (ms)')
        plt.title(f"{pct}% Update")
        plt.xlabel("Interval"); plt.ylabel("Value")
        plt.legend(); plt.grid()
        pdf.savefig(); plt.close()

    # final summary plot
    plt.figure(figsize=(10,5))
    plt.plot(summary['pct'], summary['qps'], '-o', label='QPS')
    plt.plot(summary['pct'], summary['lat'], '-s', label='Latency')
    plt.plot(summary['pct'], summary['rec'], '-^', label='Recall')
    plt.title("Milvus Dynamic Updates Summary")
    plt.xlabel("Update %"); plt.ylabel("Metric")
    plt.legend(); plt.grid()
    pdf.savefig(); plt.close()

    pdf.close()

if __name__ == "__main__":
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    simulate_milvus_dynamic(".", plot_dir/"milvus_dynamic.pdf")
