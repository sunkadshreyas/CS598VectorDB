import numpy as np
from annoy import AnnoyIndex
from threading import Event, Thread, Lock
import time
from pathlib import Path

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

# Metric calculation
def compute_recall(results, ground_truth, k):
    correct = 0
    for res, gt in zip(results, ground_truth):
        correct += len(set(res[:k]).intersection(set(gt[:k])))
    return correct / (len(results) * k)

def annoy_search(index, xq, topk):
    results = []
    for v in xq:
        idxs = index.get_nns_by_vector(v, topk)
        results.append(idxs)
    return results

def background_search_loop(index, xq, gt, topk, log, stop_event, lock):
    while not stop_event.is_set():
        start = time.time()
        # with lock:
        I = annoy_search(index, xq, topk)
        end = time.time()
        qps = xq.shape[0] / (end - start)
        latency = (end - start) * 1000
        recall = compute_recall(I, gt, topk)
        log['qps'].append(qps)
        log['latency'].append(latency)
        log['recall'].append(recall)
        time.sleep(1)

# Main evaluation
def simulate_dynamic_updates_annoy(root_dir, txt_path, update_percents=[50], topk=10):
    xt, xb, xq, gt = load_dataset(root_dir)

    txt_log = open(txt_path, "w")

    dim = xb.shape[1]
    base_size = xb.shape[0]

    # Build initial Annoy index
    index = AnnoyIndex(dim, metric='euclidean')
    for i in range(base_size):
        index.add_item(i, xb[i])
    index.build(100)

    start = time.time()
    I = annoy_search(index, xq, topk)
    end = time.time()
    baseline_qps = xq.shape[0] / (end - start)
    baseline_latency = (end - start) * 1000
    baseline_recall = compute_recall(I, gt, topk)

    print(f"\nBaseline - QPS: {baseline_qps:.2f}, Latency: {baseline_latency:.2f}ms, Recall: {baseline_recall:.4f}")

    results_summary = {
        'update_percent': [0],
        'final_qps': [baseline_qps],
        'final_latency': [baseline_latency],
        'final_recall': [baseline_recall]
    }

    for update_percent in update_percents:
        print(f"\nRunning with {update_percent}% updates...")
        num_updates = int(base_size * update_percent / 100)

        log = {'qps': [], 'latency': [], 'recall': []}
        lock = Lock()
        stop_event = Event()
        search_thread = Thread(target=background_search_loop, args=(index, xq, gt, topk, log, stop_event, lock))
        search_thread.start()

        time.sleep(30)

        # with lock:
        start_delete = time.time()
        log['qps'].append(-1)
        log['latency'].append(-1)
        log['recall'].append(-1)
        index = AnnoyIndex(dim, metric='euclidean')
        for i in range(base_size * update_percent // 100):
            index.add_item(i, xb[i])
        index.build(100)
        delete_latency = time.time() - start_delete
        log['qps'].append(-2)
        log['latency'].append(-2)
        log['recall'].append(-2)
        print(f"Delete latency: {delete_latency:.4f}s")

        start_insert = time.time()
        log['qps'].append(-3)
        log['latency'].append(-3)
        log['recall'].append(-3)
        index2 = AnnoyIndex(dim, metric='euclidean')
        for i in range(base_size):     
            index2.add_item(i, xb[i])
        index2.build(100)
        insert_latency = time.time() - start_insert
        log['qps'].append(-4)
        log['latency'].append(-4)
        log['recall'].append(-4)
        print(f"Insert latency: {insert_latency:.4f}s")

        time.sleep(30)
        stop_event.set()
        search_thread.join()

        avg_qps = np.mean(log['qps'][-5:])
        avg_latency = np.mean(log['latency'][-5:])
        avg_recall = np.mean(log['recall'][-5:])

        results_summary['update_percent'].append(update_percent)
        results_summary['final_qps'].append(avg_qps)
        results_summary['final_latency'].append(avg_latency)
        results_summary['final_recall'].append(avg_recall)

        txt_log.write(f"\n--- {update_percent}% Update ---\n")
        for i, (qps, latency, recall) in enumerate(zip(log['qps'], log['latency'], log['recall'])):
            txt_log.write(f"Interval {i+1}: QPS = {qps:.2f} queries/sec, Latency = {latency:.2f} ms, Recall = {recall:.4f}\n")

    txt_log.close()

# --- Main ---
if __name__ == "__main__":
    plot_dir = Path("logs")
    plot_dir.mkdir(parents=True, exist_ok=True)
    txt_path = plot_dir / "dynamic_updates_annoy.txt"
    simulate_dynamic_updates_annoy(".", txt_path)
