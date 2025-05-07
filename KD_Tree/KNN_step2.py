import heapq
import math

def euclidean_distance(p1, p2):
    # Tính khoảng cách Euclidean giữa hai điểm (không tính nhãn)
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1) - 1)))

def knn_search(kdtree, query_point, k):
    # Sử dụng heap để lưu k điểm gần nhất
    best_points = []  # heap lưu trữ (-khoảng_cách, điểm)
    
    def search(node, depth=0):
        if node is None:
            return
        
        # Tính khoảng cách từ query_point đến điểm hiện tại
        distance = euclidean_distance(query_point, node.point)
        
        # Nếu heap chưa đủ k phần tử hoặc khoảng cách hiện tại nhỏ hơn khoảng cách lớn nhất trong heap
        if len(best_points) < k:
            # Thêm vào với khoảng cách âm để tạo min-heap
            heapq.heappush(best_points, (-distance, node.point))
        elif -best_points[0][0] > distance:
            # Thay thế điểm xa nhất nếu tìm thấy điểm gần hơn
            heapq.heappushpop(best_points, (-distance, node.point))
        
        # Xác định trục phân chia tại độ sâu hiện tại
        k = len(query_point) - 1
        axis = depth % k
        
        # Xác định nhánh chính và nhánh thay thế
        if query_point[axis] < node.point[axis]:
            primary, alternative = node.left, node.right
        else:
            primary, alternative = node.right, node.left
        
        # Duyệt nhánh chính trước
        search(primary, depth + 1)
        
        # Kiểm tra xem có cần duyệt nhánh thay thế không
        # Nếu heap chưa đủ k phần tử hoặc khoảng cách đến đường phân chia nhỏ hơn khoảng cách lớn nhất trong heap
        if len(best_points) < k or abs(query_point[axis] - node.point[axis]) < -best_points[0][0]:
            search(alternative, depth + 1)
    
    # Bắt đầu tìm kiếm từ gốc
    search(kdtree)
    
    # Chuyển từ heap sang danh sách kết quả
    result = [point for _, point in sorted(best_points, key=lambda x: -x[0])]
    return result

def classify(kdtree, point, k):
    # Tìm k lân cận gần nhất
    neighbors = knn_search(kdtree, point, k)
    
    # Đếm số lượng nhãn
    labels = {}
    for neighbor in neighbors:
        label = neighbor[-1]  # Nhãn là phần tử cuối cùng
        labels[label] = labels.get(label, 0) + 1
    
    # Trả về nhãn có số lượng nhiều nhất
    return max(labels.items(), key=lambda x: x[1])[0]
