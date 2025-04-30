import faiss
import numpy as np
from threading import Event, Thread, Lock
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
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

def background_search_loop(index, xq, gt, topk, log, stop_event, lock):
    while not stop_event.is_set():
        start = time.time()
        with lock:
            D, I = index.search(xq, topk)
        end = time.time()
        qps = xq.shape[0] / (end - start)
        latency = (end - start) * 1000
        recall = compute_recall(I, gt, topk)
        log['qps'].append(qps)
        log['latency'].append(latency)
        log['recall'].append(recall)
        time.sleep(0.5)

# Main evaluation
def simulate_dynamic_updates_simple(root_dir, pdf_path, update_percents=[25, 75], topk=10):
    xt, xb, xq, gt = load_dataset(root_dir)

    pdf = PdfPages(pdf_path)
    txt_path = str(pdf_path).replace(".pdf", ".txt")
    txt_log = open(txt_path, "w")

    base_size = xb.shape[0]

    index = faiss.IndexHNSWFlat(xb.shape[1], 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 64
    index = faiss.IndexIDMap(index)
    index.add_with_ids(xb, np.arange(xb.shape[0]))

    start = time.time()
    D, I = index.search(xq, topk)
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

        time.sleep(2)

        with lock:
            start_del = time.time()
            index = faiss.IndexHNSWFlat(xb.shape[1], 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 64
            index = faiss.IndexIDMap(index)
            index.add_with_ids(xb[:base_size - num_updates], np.arange(base_size - num_updates))
            delete_latency = time.time() - start_del
            print(f"Delete latency: {delete_latency:.4f}s")

        with lock:
            start_ins = time.time()
            index.add_with_ids(xb[base_size - num_updates:], np.arange(base_size - num_updates, base_size))
            insert_latency = time.time() - start_ins
            print(f"Insert throughput: {num_updates / insert_latency:.2f} vectors/sec")

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
            txt_log.write(f"Interval {i+1}: QPS = {qps:.2f} queries/sec, Latency = {latency:.2f} ms, Recall = {recall:.4f}\n")

        plt.figure(figsize=(10, 4))
        plt.plot(log['qps'], label='QPS')
        plt.plot(log['latency'], label='Latency (ms)')
        plt.xlabel("Time Interval")
        plt.ylabel("Value")
        plt.title(f"{update_percent}% Update - QPS & Latency")
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
    plt.title("QPS, Latency, Recall vs. Update Percent")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()

    pdf.close()
    txt_log.close()


# --- Main ---
if __name__ == "__main__":
    
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_dir / "dynamic_updates.pdf"
    simulate_dynamic_updates_simple(".", pdf_path)
