class KDNode:
    def __init__(self, point, split_axis=0, left=None, right=None):
        self.point = point
        self.split_axis = split_axis
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if not points:
        return None
    
    # Xác định trục phân chia (xoay vòng qua các chiều)
    k = len(points[0]) - 1  # Trừ 1 vì điểm cuối cùng là nhãn
    axis = depth % k
    
    # Sắp xếp điểm theo trục phân chia hiện tại
    points.sort(key=lambda x: x[axis])
    
    # Tìm điểm trung vị
    median = len(points) // 2
    
    # Tạo nút với điểm trung vị
    node = KDNode(
        point=points[median],
        split_axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )
    
    return node
