import faiss
import numpy as np
from threading import Event, Thread
import time
import matplotlib.pyplot as plt

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

# Time-series plotting
def plot_time_series(metrics_log):
    for metric_name, values in metrics_log.items():
        times, metrics = zip(*values)
        plt.plot(times, metrics, label=metric_name)
    plt.xlabel("Time (s)")
    plt.ylabel("Metric Value")
    plt.title("Search Metrics Over Time During Dynamic Updates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Search thread
def search_worker(index, xq, gt, topk, log, stop_event):
    while not stop_event.is_set():
        start = time.time()
        D, I = index.search(xq, topk)
        end = time.time()
        latency = (end - start) * 1000
        qps = xq.shape[0] / (end - start)
        recall = (I[:, :topk] == gt[:, :topk]).sum() / (topk * len(gt))
        log['latency'].append(latency)
        log['qps'].append(qps)
        log['recall'].append(recall)
        time.sleep(0.1)

# Main evaluation
def simulate_dynamic_updates_simple(root_dir, update_percents=[25, 75], topk=10):
    xt, xb, xq, gt = load_dataset(root_dir)

    base_size = 100_000
    update_pool = xb[base_size:base_size + 500_000]
    baseline_data = xb[:base_size]

    results_summary = {
        'update_percent': [],
        'final_qps': [],
        'final_latency': [],
        'final_recall': []
    }

    # === Baseline ===
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(baseline_data)
    print("Running baseline...")

    start = time.time()
    D, I = index.search(xq, topk)
    end = time.time()
    baseline_latency = (end - start) * 1000
    baseline_qps = xq.shape[0] / (end - start)
    baseline_recall = (I[:, :topk] == gt[:, :topk]).sum() / (topk * len(gt))
    print(f"\n[Baseline] QPS: {baseline_qps:.2f}, Latency: {baseline_latency:.2f}ms, Recall: {baseline_recall:.4f}")

    # Save baseline for final plot
    results_summary['update_percent'].append(0)
    results_summary['final_qps'].append(baseline_qps)
    results_summary['final_latency'].append(baseline_latency)
    results_summary['final_recall'].append(baseline_recall)

    # === Dynamic Updates ===
    for percent in update_percents:
        print(f"\n>> Update Percent = {percent}%")
        num_updates = int(base_size * percent / 100)

        # Build original index
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(baseline_data)

        log = {'qps': [], 'latency': [], 'recall': []}
        stop_event = Event()
        search_thread = Thread(target=search_worker, args=(index, xq, gt, topk, log, stop_event))
        search_thread.start()

        time.sleep(1)  # Let search thread warm up

        # Simulate delete + insert
        remaining = baseline_data[num_updates:]
        new = update_pool[:num_updates]
        updated_data = np.vstack([remaining, new])

        start_del = time.time()
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(updated_data)
        insert_latency = time.time() - start_del

        time.sleep(2.5)  # Let search thread run a bit more
        stop_event.set()
        search_thread.join()

        # Metrics
        final_qps = np.mean(log['qps'])
        p99_latency = np.percentile(log['latency'], 99)
        final_recall = np.mean(log['recall'])
        print(f"[{percent}%] QPS: {final_qps:.2f}, p99 Latency: {p99_latency:.2f}ms, Recall: {final_recall:.4f}")

        # Append results
        results_summary['update_percent'].append(percent)
        results_summary['final_qps'].append(final_qps)
        results_summary['final_latency'].append(p99_latency)
        results_summary['final_recall'].append(final_recall)

        # === Time-series plot for this update percent ===
        plt.figure(figsize=(10, 4))
        plt.plot(log['qps'], label='QPS')
        plt.plot(log['latency'], label='Latency (ms)')
        plt.xlabel("Time (interval)")
        plt.ylabel("Value")
        plt.title(f"Time-series QPS/Latency during {percent}% Update")
        plt.legend()
        plt.grid()
        plt.show()

    # === Summary plot ===
    update_percents_full = results_summary['update_percent']
    plt.figure(figsize=(12, 6))

    plt.plot(update_percents_full, results_summary['final_qps'], marker='o', label='QPS')
    plt.plot(update_percents_full, results_summary['final_latency'], marker='s', label='p99 Latency (ms)')
    plt.plot(update_percents_full, results_summary['final_recall'], marker='^', label='Recall')

    plt.xlabel("Update Percent")
    plt.ylabel("Metric Value")
    plt.title("Summary: QPS, Latency, and Recall vs. Update Percent")
    plt.legend()
    plt.grid()
    plt.show()

def plot_dynamic_metrics(baseline_results, dynamic_results):

    # Time-series plots for each update_percent
    for result in dynamic_results:
        percent = result["percent"]
        metrics = result["metrics"]
        timestamps = [t - metrics["timestamps"][0] for t in metrics["timestamps"]]

        plt.figure(figsize=(16, 5))
        plt.suptitle(f"Search Performance During {percent}% Update")

        # Latency
        plt.subplot(1, 3, 1)
        plt.plot(timestamps, metrics["latencies"], label="Search Latency (ms)")
        plt.axhline(y=baseline_results["latency"], color='r', linestyle='--', label='Baseline')
        plt.xlabel("Time (s)")
        plt.ylabel("Latency (ms)")
        plt.legend()
        plt.grid()

        # QPS
        plt.subplot(1, 3, 2)
        plt.plot(timestamps, metrics["qps"], label="QPS")
        plt.axhline(y=baseline_results["qps"], color='r', linestyle='--', label='Baseline')
        plt.xlabel("Time (s)")
        plt.ylabel("QPS")
        plt.legend()
        plt.grid()

        # Recall
        plt.subplot(1, 3, 3)
        plt.plot(timestamps, metrics["recalls"], label="Recall@k")
        plt.axhline(y=baseline_results["recall"], color='r', linestyle='--', label='Baseline')
        plt.xlabel("Time (s)")
        plt.ylabel("Recall")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    # Summary bar charts for insert/delete latency
    percents = [r["percent"] for r in dynamic_results]
    insert_latencies = [r["metrics"]["insert_latency"] for r in dynamic_results]
    delete_latencies = [r["metrics"]["delete_latency"] for r in dynamic_results]

    plt.figure(figsize=(10, 4))
    plt.bar(percents, insert_latencies, width=4, label="Insert Latency (ms)")
    plt.bar(percents, delete_latencies, width=4, bottom=insert_latencies, label="Delete Latency (ms)")
    plt.xlabel("Update Percent (%)")
    plt.ylabel("Latency (ms)")
    plt.title("Insert/Delete Latency vs Update Load")
    plt.legend()
    plt.grid()
    plt.show()


# --- Main ---
if __name__ == "__main__":
    simulate_dynamic_updates_simple(".")
