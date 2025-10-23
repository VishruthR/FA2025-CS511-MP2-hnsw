import faiss
import h5py
import numpy as np
import time
import os
import requests
import matplotlib.pyplot as plt

def download_data(url, data_path):
    # download sift dataset
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(data_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def evaluate_hnsw():
    # download data, build index, run query
    url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    data_path = "sift.data.hd5"

    # download sift dataset
    # response = requests.get(url, stream=True)
    # response.raise_for_status()
    # with open(data_path, "wb") as f:
    #     for chunk in response.iter_content(chunk_size=8192):
    #         if chunk:
    #             f.write(chunk)
    
    # grab test and train datasets
    with h5py.File(data_path, "r") as f:
        # print("HDF5 File Structure:")
        # f.visititems(print_h5_structure)

        train_set = f["train"][:]
        test_set = f["test"][:]

    # Initialize hnsw
    d = 128
    M = 16
    hnsw = faiss.IndexHNSWFlat(d, M)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 200

    hnsw.add(train_set)

    test_vec = test_set[0][np.newaxis, :]
    distances, indices = hnsw.search(test_vec, k=10)

    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    with open("output.txt", "w") as f:
        out_string = "\n".join([str(i) for i in indices[0]])
        f.write(out_string)
    

def hnsw_vs_lsh():
    # Assume data already downloaded
    data_path = "sift.data.hd5"

    # grab test and train datasets
    with h5py.File(data_path, "r") as f:
        train_set = f["train"][:]
        test_set = f["test"][:]
        test_neighbors = f["neighbors"][:]

    def evaluate_model(index):
        """
        Accepts a vector index and tests its performance on test_set, measuring 1-Recall@1 and QPS
        """
        total_examples = len(test_set)
        start_time = time.time()
        _, neighbors = index.search(test_set, k=1)
        end_time = time.time()

        num_correct = 0
        for pred, truth in zip(neighbors, test_neighbors):
            # check if prediction matches nearest neighbor
            if pred == truth[0]:
                num_correct += 1
 
        print(f"Evaluated {total_examples} in {end_time - start_time} seconds")
        print(f"Guessed {num_correct} NN correctly")
        time_elapsed = end_time - start_time
        return (total_examples, time_elapsed, num_correct)

    d = 128
    M = 32
    efSearch_values = [10, 50, 100, 200]
    hnsw_qps = []
    hnsw_recall = []
    print("HNSW")
    for efSearch in efSearch_values:
        print("--------------------------------------------")
        print(f"Testing efSearch: {efSearch}")
        hnsw = faiss.IndexHNSWFlat(d, M)
        hnsw.hnsw.efConstruction = 200
        hnsw.hnsw.efSearch = efSearch

        hnsw.add(train_set)

        (n, time_elapsed, num_correct) = evaluate_model(hnsw)
        hnsw_qps.append(n / time_elapsed)
        hnsw_recall.append(num_correct / n)

    nbits_values = [32, 64, 512, 768]
    lsh_qps = []
    lsh_recall = []
    print("LSH")
    for nbits in nbits_values:
        print("--------------------------------------------")
        print(f"Testing nbits: {nbits}")
        lsh = faiss.IndexLSH(d, nbits)
        lsh.add(train_set)

        (n, time_elapsed, num_correct) = evaluate_model(lsh)
        lsh_qps.append(n / time_elapsed)
        lsh_recall.append(num_correct / n)
        
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel("1-Recall@1")
    ax.set_ylabel("QPS")
    ax.scatter(hnsw_recall, hnsw_qps, color="red", label="HNSW")
    ax.plot(hnsw_recall, hnsw_qps, color="red")
    for (xi, yi, label) in zip(hnsw_recall, hnsw_qps, efSearch_values):
        plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(3, 3), 
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.3))

    ax.scatter(lsh_recall, lsh_qps, color="blue", label="LSH")
    ax.plot(lsh_recall, lsh_qps, color="blue")
    for (xi, yi, label) in zip(lsh_recall, lsh_qps, nbits_values):
        plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(3, 3),
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.3))

    plt.title("QPS vs. 1-Recall@1")

    plt.legend()
    plt.savefig("HNSW-vs-LSH.png")

def evaluate_hnsw_size():
    urls = ["http://ann-benchmarks.com/sift-128-euclidean.hdf5", "http://ann-benchmarks.com/mnist-784-euclidean.hdf5", "http://ann-benchmarks.com/lastfm-64-dot.hdf5", "http://ann-benchmarks.com/glove-100-angular.hdf5"]
    data_paths = ["sift.data.hd5", "mnist.data.hd5", "lastfm.data.hd5", "glove.data.hd5"]
    metrics = ["euclidean", "euclidean", "angular", "angular"]
    dimensions = [128, 784, 65, 100]

    # for url, path in zip(urls, data_paths):
    #     download_data(url, path)

    def evaluate_model(index, test, truth_neighbors):
        """
        Accepts a vector index and tests its performance on test, measuring 1-Recall@1 and QPS
        """
        total_examples = len(test)
        start_time = time.time()
        _, neighbors = index.search(test, k=1)
        end_time = time.time()

        num_correct = 0
        for pred, truth in zip(neighbors, truth_neighbors):
            # check if prediction matches nearest neighbor
            if pred == truth[0]:
                num_correct += 1

        time_elapsed = end_time - start_time
        return (total_examples, time_elapsed, num_correct)
    
    M = [4, 8, 12, 24, 48]
    build_times = []
    qps = []
    recalls = []

    for i in range(len(data_paths)):
        print(f"Processing dataset: {data_paths[i]}")
        temp_build_times = []
        temp_qps = []
        temp_recalls = []
        for m in M:
            metric = metrics[i]
            d = dimensions[i]

            with h5py.File(data_paths[i], "r") as f:
                train_set = f["train"][:].astype("float32").copy()
                test_set = f["test"][:].astype("float32").copy()
                test_neighbors = f["neighbors"][:]  # Keep as integers, don't convert to float32

            # normalize to calculate cosine sim with inner product
            if metric == "angular":
                faiss.normalize_L2(train_set)
                faiss.normalize_L2(test_set)

            hnswMetric = faiss.METRIC_L2
            if metric == "angular":
                hnswMetric = faiss.METRIC_INNER_PRODUCT
            hnsw = faiss.IndexHNSWFlat(d, m)
            hnsw.metric_type = hnswMetric
            hnsw.hnsw.efConstruction = 200
            hnsw.hnsw.efSearch = 50 # chose because higher efSearch values only offer marginal gains to recall

            startBuildTime = time.time()
            hnsw.add(train_set)
            totalBuildTime = time.time() - startBuildTime

            (n, eval_time, num_correct) = evaluate_model(hnsw, test_set, test_neighbors)

            temp_build_times.append(totalBuildTime)
            temp_qps.append(n / eval_time)
            temp_recalls.append(num_correct / n)
        
        build_times.append(temp_build_times)
        qps.append(temp_qps)
        recalls.append(temp_recalls)

    print("build times: ", build_times)
    print("qps: ", qps)
    print("recalls: ", recalls)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel("1-Recall@1")
    ax.set_ylabel("QPS")
    
    colors = ["red", "blue", "green", "orange"]
    for i in range(len(data_paths)):
        color = colors[i % len(colors)]
        dataset_name = data_paths[i].split(".")[0]
        ax.scatter(recalls[i], qps[i], color=color, label=dataset_name)
        ax.plot(recalls[i], qps[i], color=color)
        offset = (3, 3)
        if dataset_name == "mnist":
            offset = (7, 7)
        for (xi, yi, label) in zip(recalls[i], qps[i], M):
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=offset, 
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.3))

    plt.title("QPS vs. 1-Recall@1")

    plt.legend()
    plt.savefig("HNSW-size-1.png")

    plt.close()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel("1-Recall@1")
    ax.set_ylabel("Index Build Time (s)")
    
    colors = ["red", "blue", "green", "orange"]
    for i in range(len(data_paths)):
        color = colors[i % len(colors)]
        dataset_name = data_paths[i].split(".")[0]
        ax.scatter(recalls[i], build_times[i], color=color, label=dataset_name)
        ax.plot(recalls[i], build_times[i], color=color)
        offset = (3, 3)
        if dataset_name == "mnist":
            offset = (7, 7)
        for (xi, yi, label) in zip(recalls[i], build_times[i], M):
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=offset, 
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.3))

    plt.title("Index build times vs. 1-Recall@1")

    plt.legend()
    plt.savefig("HNSW-size-2.png")
    plt.close()

def hnsw_vs_diskann():
        # Assume data already downloaded
    data_path = "sift.data.hd5"

    # grab test and train datasets
    with h5py.File(data_path, "r") as f:
        train_set = f["train"][:]
        test_set = f["test"][:]
        test_neighbors = f["neighbors"][:]

    def evaluate_model(index):
        """
        Accepts a vector index and tests its performance on test_set, measuring 1-Recall@1 and QPS
        """
        total_examples = len(test_set)
        pred_neighbors = []
        latencies = []
        for i in range(total_examples):
            start_time = time.time()
            _, neighbors = index.search(test_set[i].reshape(1, -1), k=1)
            end_time = time.time()
            latencies.append(end_time - start_time)
            pred_neighbors.append(neighbors[0])
            
        num_correct = 0
        for pred, truth in zip(pred_neighbors, test_neighbors):
            # check if prediction matches nearest neighbor
            if pred == truth[0]:
                num_correct += 1
 
        return (total_examples, np.average(latencies) * 1000, num_correct)
    
    # for HNSW we will vary efConstruction and efSearch
    print("HNSW")
    d = 128
    M = 32
    efSearch_values = [10, 50, 100, 200]
    M_values = [4, 8, 12, 24, 48]

    hnsw_recall_efSearch = []
    hnsw_latency_efSearch = []
    for efSearch in efSearch_values:
        print("--------------------------------------------")
        print(f"Testing efSearch: {efSearch}")
        hnsw = faiss.IndexHNSWFlat(d, M)
        hnsw.hnsw.efConstruction = 100
        hnsw.hnsw.efSearch = efSearch

        hnsw.add(train_set)

        (n, avg_latency, num_correct) = evaluate_model(hnsw)
        hnsw_latency_efSearch.append(avg_latency)
        hnsw_recall_efSearch.append(num_correct / n)

    hnsw_recall_M = []
    hnsw_latency_M = []
    for M_val in M_values:
        print("--------------------------------------------")
        print(f"Testing M: {M_val}")
        hnsw = faiss.IndexHNSWFlat(d, M_val)
        hnsw.hnsw.efConstruction = 100
        hnsw.hnsw.efSearch = 100

        hnsw.add(train_set)

        (n, avg_latency, num_correct) = evaluate_model(hnsw)
        hnsw_latency_M.append(avg_latency)
        hnsw_recall_M.append(num_correct / n)

    print(hnsw_latency_efSearch)
    print(hnsw_recall_efSearch)

    print(hnsw_latency_M)
    print(hnsw_recall_M)



if __name__ == "__main__":
    # evaluate_hnsw()
    # hnsw_vs_lsh()
    # evaluate_hnsw_size()
    hnsw_vs_diskann()
