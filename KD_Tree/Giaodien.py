import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time
import random
import math
import heapq

# KD-Tree logic
class KDNode:
    def __init__(self, point, split_axis=0, left=None, right=None):
        self.point = point
        self.split_axis = split_axis
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if not points:
        return None
    k = len(points[0]) - 1
    axis = depth % k
    points.sort(key=lambda x: x[axis])
    median = len(points) // 2
    return KDNode(
        point=points[median],
        split_axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1) - 1)))

def knn_search(kdtree, query_point, k):
    best_points = []
    def search(node, depth=0):
        if node is None: return
        distance = euclidean_distance(query_point, node.point)
        if len(best_points) < k:
            heapq.heappush(best_points, (-distance, node.point))
        elif -best_points[0][0] > distance:
            heapq.heappushpop(best_points, (-distance, node.point))
        axis = depth % (len(query_point) - 1)
        primary, alternative = (node.left, node.right) if query_point[axis] < node.point[axis] else (node.right, node.left)
        search(primary, depth + 1)
        if len(best_points) < k or abs(query_point[axis] - node.point[axis]) < -best_points[0][0]:
            search(alternative, depth + 1)
    search(kdtree)
    return [point for _, point in sorted(best_points, key=lambda x: -x[0])]

def classify(kdtree, point, k):
    neighbors = knn_search(kdtree, point, k)
    labels = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        labels[label] = labels.get(label, 0) + 1
    return max(labels.items(), key=lambda x: x[1])[0]

def brute_force_knn(data, query_point, k):
    distances = [(euclidean_distance(point, query_point), point) for point in data]
    return [point for _, point in sorted(distances)[:k]]

def measure_execution_time(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, end - start

def generate_test_data(n_samples, n_features, n_classes):
    return [[random.uniform(0, 100) for _ in range(n_features)] + [random.randint(0, n_classes - 1)] for _ in range(n_samples)]

# GUI class
class KNNGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("So sánh KNN với KD-Tree và Brute Force")
        
        self.text = scrolledtext.ScrolledText(master, width=100, height=25)
        self.text.pack()

        self.btn_compare = tk.Button(master, text="So sánh hiệu suất", command=self.compare_performance)
        self.btn_compare.pack(pady=5)

        self.btn_classify = tk.Button(master, text="Phân loại điểm mới", command=self.classify_point)
        self.btn_classify.pack(pady=5)

    def compare_performance(self):
        output = "| Kích thước dữ liệu | Thời gian xây dựng KD-Tree | Thời gian tìm kiếm KD-Tree | Thời gian tìm kiếm Brute Force |\n"
        output += "|---------------------|----------------------------|---------------------------|--------------------------------|\n"

        data_sizes = [1000, 5000, 10000]
        for size in data_sizes:
            data = generate_test_data(size, 3, 3)
            test_points = generate_test_data(10, 3, 3)

            kdtree, build_time = measure_execution_time(build_kdtree, data.copy())

            kdtree_search_time = sum(measure_execution_time(knn_search, kdtree, pt, 5)[1] for pt in test_points) / 10
            brute_force_time = sum(measure_execution_time(brute_force_knn, data, pt, 5)[1] for pt in test_points) / 10

            output += f"| {size:<19} | {build_time:<26.6f} | {kdtree_search_time:<25.6f} | {brute_force_time:<30.6f} |\n"

        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, output)

    def classify_point(self):
        data = generate_test_data(1000, 2, 3)
        kdtree = build_kdtree(data.copy())
        new_point = [50, 50, None]
        label = classify(kdtree, new_point, 5)
        self.text.insert(tk.END, f"\nNhãn dự đoán cho điểm {new_point[:-1]} là: {label}\n")

        # Plot
        fig, ax = plt.subplots()
        colors = ['red', 'green', 'blue']
        for point in data:
            ax.scatter(point[0], point[1], color=colors[point[-1]], alpha=0.5)
        ax.scatter(new_point[0], new_point[1], color='black', marker='x', s=100, label='Điểm mới')
        ax.legend()
        ax.set_title("Phân loại bằng KD-Tree")

        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.get_tk_widget().pack()
        canvas.draw()

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = KNNGUI(root)
    root.mainloop()
