import time
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors

# Hàm tạo dữ liệu thử nghiệm
def generate_test_data(size, n_features, n_classes):
    np.random.seed(42)
    X = np.random.rand(size, n_features)
    y = np.random.randint(0, n_classes, size)
    return X

# Hàm đo thời gian thực thi
def measure_execution_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return result, end_time - start_time

# Xây dựng KD-Tree
def build_kdtree(data):
    kdtree = KDTree(data)
    return kdtree

# Tìm k láng giềng gần nhất bằng KD-Tree
def knn_search(kdtree, point, k):
    distances, indices = kdtree.query(point, k)
    return indices

# Tìm k láng giềng gần nhất bằng Brute Force
def brute_force_knn(data, point, k):
    distances = np.sqrt(np.sum((data - point)**2, axis=1))
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices

# Thực hiện so sánh
def compare_performance():
    # Tạo dữ liệu thử nghiệm
    data_sizes = [1000, 5000, 10000]
    n_features = 3
    n_classes = 3
    k = 5

    results = []

    for size in data_sizes:
        data = generate_test_data(size, n_features, n_classes)
        test_points = generate_test_data(10, n_features, n_classes)

        # Đo thời gian xây dựng KD-Tree
        kdtree, build_time = measure_execution_time(build_kdtree, data.copy())

        # Đo thời gian tìm kiếm với KD-Tree
        kdtree_search_times = []
        for point in test_points:
            _, time_taken = measure_execution_time(knn_search, kdtree, point, k)
            kdtree_search_times.append(time_taken)
        kdtree_search_time = np.mean(kdtree_search_times)

        # Đo thời gian tìm kiếm với Brute Force
        bf_search_times = []
        for point in test_points:
            _, time_taken = measure_execution_time(brute_force_knn, data, point, k)
            bf_search_times.append(time_taken)
        bf_search_time = np.mean(bf_search_times)

        results.append({
            'size': size,
            'kdtree_build_time': build_time,
            'kdtree_search_time': kdtree_search_time,
            'brute_force_search_time': bf_search_time
        })

    return results

if __name__ == "__main__":
    comparison_results = compare_performance()
    for result in comparison_results:
        print(f"Kích thước dữ liệu: {result['size']}")
        print(f"  Thời gian xây dựng KD-Tree: {result['kdtree_build_time']:.6f} giây")
        print(f"  Thời gian tìm kiếm trung bình (KD-Tree): {result['kdtree_search_time']:.6f} giây")
        print(f"  Thời gian tìm kiếm trung bình (Brute Force): {result['brute_force_search_time']:.6f} giây")
        print("-" * 30)