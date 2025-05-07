import time
import random
import math
from memory_profiler import profile

# Hàm tính khoảng cách Euclidean
def euclidean_distance(p1, p2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

# Phương pháp vét cạn (Brute Force)
def brute_force_knn(data, query_point, k):
    distances = [(euclidean_distance(point[:-1], query_point), point) for point in data]
    return [point for _, point in sorted(distances)[:k]]

# Hàm đo thời gian thực thi
def measure_execution_time(func, *args):
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    return result, end_time - start_time

# Hàm tạo dữ liệu thử nghiệm ngẫu nhiên
def generate_test_data(n_samples, n_features, n_classes):
    data = []
    for _ in range(n_samples):
        features = [random.uniform(0, 100) for _ in range(n_features)]
        label = random.randint(0, n_classes - 1)
        data.append(features + [label])
    return data

# Dummy build_kdtree chỉ để chạy được (bạn thay bằng KD-Tree thật nhé)
def build_kdtree(data):
    # Giả sử KD-Tree chỉ cần lưu lại data (bạn thay bằng KDTree của bạn)
    return data

# Đo bộ nhớ khi xây KD-Tree
@profile
def test_kdtree_memory_usage(data):
    return build_kdtree(data.copy())

# Đo bộ nhớ khi Brute-Force KNN
@profile
def test_brute_force_memory_usage(data, query_point, k):
    return brute_force_knn(data.copy(), query_point, k)

# -----------------------------------------
if __name__ == "__main__":
    n_samples = 10000
    n_features = 5
    n_classes = 3
    k = 5

    data = generate_test_data(n_samples, n_features, n_classes)
    query_point = [random.uniform(0, 100) for _ in range(n_features)]

    print("=== Đo thời gian Brute Force ===")
    _, bf_time = measure_execution_time(brute_force_knn, data, query_point, k)
    print(f"Brute Force KNN mất {bf_time:.4f} giây")

    print("\n=== Đo memory KD-Tree ===")
    test_kdtree_memory_usage(data)

    print("\n=== Đo memory Brute Force ===")
    test_brute_force_memory_usage(data, query_point, k)
