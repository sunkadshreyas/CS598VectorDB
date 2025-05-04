import ngtpy
import numpy as np
from threading import Event, Thread, Lock
import time
import shutil
from pathlib import Path
import os

# Dataset loading
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

def compute_recall(results, ground_truth, k):
    correct = 0
    for res, gt in zip(results, ground_truth):
        correct += len(set(res[:k]).intersection(set(gt[:k])))
    return correct / (len(results) * k)

def search_ngt(index, xq, topk):
    return [list(map(lambda x: x[0], index.search(vec, topk))) for vec in xq]

def background_search_loop(index, xq, gt, topk, log, stop_event, lock):
    while not stop_event.is_set():
        start = time.time()
        with lock:
            I = search_ngt(index, xq, topk)
        end = time.time()
        qps = xq.shape[0] / (end - start)
        latency = (end - start) * 1000
        recall = compute_recall(I, gt, topk)
        log['qps'].append(qps)
        log['latency'].append(latency)
        log['recall'].append(recall)
        time.sleep(0.25)

def simulate_dynamic_updates_ngt(root_dir, txt_path, update_percents=[25, 75], topk=10):
    xt, xb, xq, gt = load_dataset(root_dir)

    txt_log = open(txt_path, "w")

    base_size = xb.shape[0]
    dim = xb.shape[1]
    index_path = "tmp_ngt"

    def build_ngt_index(data):
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        ngtpy.create(index_path, dim)
        index = ngtpy.Index(index_path)
        for vec in data:
            index.insert(vec)
        index.build_index()
        return index

    index = build_ngt_index(xb)
    start = time.time()
    I = search_ngt(index, xq, topk)
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

        time.sleep(5)

        # with lock:
        start_del = time.time()
        log['qps'].append(-1)
        log['latency'].append(-1)
        log['recall'].append(-1)
        index = build_ngt_index(xb[:base_size - num_updates])
        delete_latency = time.time() - start_del
        log['qps'].append(-2)
        log['latency'].append(-2)
        log['recall'].append(-2)
        print(f"Delete latency: {delete_latency:.4f}s")

        # with lock:
        start_ins = time.time()
        log['qps'].append(-3)
        log['latency'].append(-3)
        log['recall'].append(-3)
        for vec in xb[base_size - num_updates:]:
            index.insert(vec)
        index.build_index()
        insert_latency = time.time() - start_ins
        log['qps'].append(-4)
        log['latency'].append(-4)
        log['recall'].append(-4)
        print(f"Insert latency: {insert_latency:.4f}s")
        print(f"Insert throughput: {num_updates / insert_latency:.2f} items/s")

        time.sleep(5)
        stop_event.set()
        search_thread.join()

        results_summary['update_percent'].append(update_percent)
        avg_qps = np.mean(log['qps'][-5:])
        avg_latency = np.mean(log['latency'][-5:])
        avg_recall = np.mean(log['recall'][-5:])
        results_summary['final_qps'].append(avg_qps)
        results_summary['final_latency'].append(avg_latency)
        results_summary['final_recall'].append(avg_recall)

        txt_log.write(f"\n--- {update_percent}% Update ---\n")
        for i, (qps, latency, recall) in enumerate(zip(log['qps'], log['latency'], log['recall'])):
            txt_log.write(f"Interval {i+1}: QPS = {qps:.2f}, Latency = {latency:.2f} ms, Recall = {recall:.4f}\n")

    txt_log.close()
    shutil.rmtree(index_path)

if __name__ == "__main__":
    log_dir = Path("logs_ngt")
    log_dir.mkdir(parents=True, exist_ok=True)
    txt_path = log_dir / "dynamic_updates_ngt.txt"
    simulate_dynamic_updates_ngt(".", txt_path)
