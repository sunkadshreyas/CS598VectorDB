import faiss
from annoy import AnnoyIndex
import hnswlib
import mrpt
import ngtpy
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import os
from tqdm import tqdm
import random

# Load SIFT1M dataset
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_dataset(root_dir):
    folder_path = f"{root_dir}/sift"
    if os.path.exists(folder_path):
        xt = fvecs_read(f"{folder_path}/sift_learn.fvecs")
        xb = fvecs_read(f"{folder_path}/sift_base.fvecs")
        xq = fvecs_read(f"{folder_path}/sift_query.fvecs")
        gt = ivecs_read(f"{folder_path}/sift_groundtruth.ivecs")
    else:
        xt = fvecs_read(f"{root_dir}/sift/sift/sift_learn.fvecs")
        xb = fvecs_read(f"{root_dir}/sift/sift/sift_base.fvecs")
        xq = fvecs_read(f"{root_dir}/sift/sift/sift_query.fvecs")
        gt = ivecs_read(f"{root_dir}/sift/sift/sift_groundtruth.ivecs")
    return xb, xq, gt, xt

def compute_recall(I, gt, topk):
    correct = 0
    for i in range(len(I)):
        pred_ids = list(map(int, I[i][:topk]))
        true_ids = list(gt[i][:topk])
        correct += len(set(pred_ids) & set(true_ids))
    return correct / (len(I) * topk)

# HNSWLib Section
def build_hnswlib(xb):
    dim = xb.shape[1]
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=xb.shape[0], ef_construction=200, M=16)
    for i in tqdm(range(xb.shape[0]), desc="Building HNSWLib Index"):
        index.add_items(xb[i], i)
    index.set_ef(64)
    return index

def search_hnswlib(index, xq, topk):
    I = index.knn_query(xq, k=topk)[0]
    return I

# MRPT Section
def build_mrpt(xb):
    index = mrpt.MRPTIndex(xb)
    index.build_autotune_sample(0.9, 10)
    return index

def search_mrpt(index, xq, topk):
    I = []
    for vec in xq:
        I.append(index.ann(vec))
    I = np.array(I)
    return I

# NGT Section
def build_ngt(xb):
    dim = xb.shape[1]
    ngtpy.create(b"tmp", dim)
    index = ngtpy.Index(b"tmp")
    for vec in xb:
        index.insert(vec)
    index.build_index()
    return index

def search_ngt(index, xq, topk):
    I = []
    for vec in xq:
        results = index.search(vec, topk)
        ids = [id for id, distance in results]
        I.append(ids)
    I = np.array(I)
    return I

# IVF-PQ Section
def build_faiss_ivfpq(xb, xt):
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(xb.shape[1]), xb.shape[1], 100, 16, 8)
    index.train(xt)
    index.add(xb)
    index.nprobe = 10
    return index

def search_faiss_ivfpq(index, xq, topk):
    D, I = index.search(xq, topk)
    return I

# ANNOY Section
def build_annoy(xb):
    dim = xb.shape[1]
    index = AnnoyIndex(dim, metric='euclidean')
    for i in range(xb.shape[0]):
        index.add_item(i, xb[i])
    index.build(50)
    return index

def search_annoy(index, xq, topk):
    results = []
    for v in xq:
        ids = index.get_nns_by_vector(v, topk)
        results.append(ids)
    return results

def evaluate_scaling_behavior(root_dir, client_counts=[10,20,30,40,50], topk=10):
    xb, xq, gt, xt = load_dataset(root_dir)
    # xb = xb[:150000]

    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    pdf_path = plot_dir / "scaling_behavior.pdf"
    pdf = PdfPages(pdf_path)

    backends = {
        "Annoy": build_annoy,
        "MRPT": build_mrpt,
        "HNSWLib": build_hnswlib,
        "NGT-ONGG": build_ngt,
        "FAISS-IVF": lambda xb: build_faiss_ivfpq(xb, xt)
    }

    all_results = {backend: {"latencies": [], "qps": [], "throughputs": [], "recalls": []} for backend in backends}

    for backend_name, build_index in backends.items():
        print(f"\n[Backend: {backend_name}] Building index...")
        index = build_index(xb)

        for num_clients in client_counts:
            print(f"[Backend: {backend_name}] Running with {num_clients} clients...")
            result_list = [None] * num_clients
            threads = []

            def single_client(client_id):
                start = time.time()
                if backend_name == "Annoy":
                    I = search_annoy(index, xq, topk)
                elif backend_name == "FAISS-IVF":
                    I = search_faiss_ivfpq(index, xq, topk)
                elif backend_name == "HNSWLib":
                    I = search_hnswlib(index, xq, topk)
                elif backend_name == "MRPT":
                    I = search_mrpt(index, xq, topk)
                elif backend_name == "NGT-ONGG":
                    I = search_ngt(index, xq, topk)
                end = time.time()
                qps = xq.shape[0] / (end - start)
                latency = (end - start) * 1000
                recall = compute_recall(I, gt, topk)
                result_list[client_id] = (qps, latency, recall)

            for i in range(num_clients):
                thread = threading.Thread(target=single_client, args=(i,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            qps_values, latency_values, recall_values = zip(*result_list)
            print(f"\n[Backend: {backend_name}] Results for {num_clients} clients: Latency: {np.mean(latency_values):.2f}ms, QPS: {np.mean(qps_values):.2f}, Recall: {np.mean(recall_values):.4f}")
            all_results[backend_name]["latencies"].append(np.mean(latency_values))
            all_results[backend_name]["qps"].append(np.mean(qps_values))
            all_results[backend_name]["throughputs"].append(np.sum(qps_values))
            all_results[backend_name]["recalls"].append(np.mean(recall_values))

    colors = {
        "Annoy": "green",
        "FAISS-IVF": "blue",
        "HNSWLib": "red",
        "MRPT": "purple",
        "NGT-ONGG": "orange"
    }

    for metric in ["latencies", "qps", "throughputs", "recalls"]:
        plt.figure(figsize=(8, 5))
        for backend_name in backends:
            plt.plot(client_counts, all_results[backend_name][metric], marker='o', label=backend_name, color=colors[backend_name])
        plt.xlabel('Number of Clients')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs Number of Clients')
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.close()

    pdf.close()
    print(f"\n📄 Scaling behavior plots saved to: {pdf_path}")

if __name__ == "__main__":
    evaluate_scaling_behavior(".")
