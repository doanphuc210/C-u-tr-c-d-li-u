import time
import random
import math
import heapq
from memory_profiler import profile

# Định nghĩa Node trong KD-Tree
class KDNode:
    def __init__(self, point, split_axis=0, left=None, right=None):
        self.point = point
        self.split_axis = split_axis
        self.left = left
        self.right = right

# Xây dựng KD-Tree
def build_kdtree(points, depth=0):
    if not points:
        return None
    
    k = len(points[0]) - 1  # Trừ 1 vì điểm cuối cùng là nhãn
    axis = depth % k
    
    points.sort(key=lambda x: x[axis])
    median = len(points) // 2
    
    node = KDNode(
        point=points[median],
        split_axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )
    
    return node

# Tính khoảng cách Euclidean
def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1) - 1)))

# Tìm kiếm K lân cận gần nhất
def knn_search(kdtree, query_point, k):
    best_points = []
    
    def search(node, depth=0):
        if node is None:
            return
        
        distance = euclidean_distance(query_point, node.point)
        
        if len(best_points) < k:
            heapq.heappush(best_points, (-distance, node.point))
        elif -best_points[0][0] > distance:
            heapq.heappushpop(best_points, (-distance, node.point))
        
        k_dim = len(query_point) - 1
        axis = depth % k_dim
        
        if query_point[axis] < node.point[axis]:
            primary, alternative = node.left, node.right
        else:
            primary, alternative = node.right, node.left
        
        search(primary, depth + 1)
        
        if len(best_points) < k or abs(query_point[axis] - node.point[axis]) < -best_points[0][0]:
            search(alternative, depth + 1)
    
    search(kdtree)
    
    result = [point for _, point in sorted(best_points, key=lambda x: -x[0])]
    return result

# Phân loại điểm dữ liệu
def classify(kdtree, point, k):
    neighbors = knn_search(kdtree, point, k)
    
    labels = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        labels[label] = labels.get(label, 0) + 1
    
    return max(labels.items(), key=lambda x: x[1])[0]

# Phương pháp vét cạn
def brute_force_knn(data, query_point, k):
    distances = [(euclidean_distance(point, query_point), point) for point in data]
    return [point for _, point in sorted(distances)[:k]]

# Đo thời gian thực thi
def measure_execution_time(func, *args):
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    return result, end_time - start_time

# Tạo dữ liệu thử nghiệm
def generate_test_data(n_samples, n_features, n_classes):
    data = []
    for _ in range(n_samples):
        features = [random.uniform(0, 100) for _ in range(n_features)]
        label = random.randint(0, n_classes - 1)
        data.append(features + [label])
    return data

# Đo lượng bộ nhớ sử dụng
@profile
def test_kdtree_memory_usage(data):
    return build_kdtree(data.copy())

@profile
def test_brute_force_memory_usage(data):
    return data.copy()

# So sánh hiệu suất
def compare_performance():

    data_sizes = [1000, 5000, 10000]
    n_features = 3
    n_classes = 3
    k = 5
    
    print("| Kích thước dữ liệu | Thời gian xây dựng KD-Tree | Thời gian tìm kiếm KD-Tree | Thời gian tìm kiếm Brute Force |")
    print("|---------------------|----------------------------|---------------------------|--------------------------------|")
    
    for size in data_sizes:
        data = generate_test_data(size, n_features, n_classes)
        test_points = generate_test_data(10, n_features, n_classes)
        
        # Đo thời gian xây dựng KD-Tree
        kdtree, build_time = measure_execution_time(build_kdtree, data.copy())
        
        # Đo thời gian tìm kiếm với KD-Tree
        kdtree_search_time = 0
        for point in test_points:
            _, time_taken = measure_execution_time(knn_search, kdtree, point, k)
            kdtree_search_time += time_taken
        kdtree_search_time /= len(test_points)
        
        # Đo thời gian tìm kiếm với Brute Force
        bf_search_time = 0
        for point in test_points:
            _, time_taken = measure_execution_time(brute_force_knn, data, point, k)
            bf_search_time += time_taken
        bf_search_time /= len(test_points)
        
        print(f"| {size:<19} | {build_time:<26.6f} | {kdtree_search_time:<25.6f} | {bf_search_time:<30.6f} |")
    
    # Đo lượng bộ nhớ sử dụng
    print("\nĐo lượng bộ nhớ sử dụng (chạy với memory_profiler):")
    print("Xem kết quả từ decorator @profile")

if __name__ == "__main__":
    print("So sánh hiệu suất giữa KNN với KD-Tree và phương pháp vét cạn:")
    compare_performance()
    
    print("\nThử nghiệm phân loại dữ liệu:")
    # Tạo dữ liệu thử nghiệm đơn giản
    data = generate_test_data(1000, 2, 3)
    kdtree = build_kdtree(data.copy())
    
    # Phân loại điểm mới
    new_point = [50, 50, None]  # None là vị trí nhãn
    predicted_label = classify(kdtree, new_point, k=5)
    print(f"Nhãn dự đoán cho điểm {new_point[:-1]}: {predicted_label}")
