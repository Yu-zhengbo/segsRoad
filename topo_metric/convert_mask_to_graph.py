import numpy as np
import cv2
import networkx as nx
from skimage.morphology import skeletonize
import sknw
from scipy.spatial import cKDTree
import numpy as np
from shapely.geometry import LineString
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import pickle
import os
from tqdm import tqdm

def mask_to_graph(mask_path_or_array):
    """
    将二值 Mask 转换为带权重的 NetworkX Graph。
    步骤：读取 -> 二值化 -> 骨架化 -> 建图
    """
    # 1. 读取和预处理
    if isinstance(mask_path_or_array, str):
        # 读取图像，确保是单通道
        mask = cv2.imread(mask_path_or_array, cv2.IMREAD_GRAYSCALE)
    else:
        mask = mask_path_or_array

    # 确保是 0/1 的二值矩阵
    mask = (mask > 127).astype(np.uint8) if np.max(mask) > 1 else mask

    # 如果掩膜全是黑的（无路），返回空图
    if np.sum(mask) == 0:
        return nx.Graph()

    # 2. 骨架化 (Skeletonize)
    # 将粗道路细化为单像素宽的线
    ske = skeletonize(mask).astype(np.uint8)

    # 3. 转换为图 (Graph Construction)
    # sknw 会自动计算边长(weight)作为几何距离
    # iso=False 表示不使用隔离点优化，保留所有连接
    graph = sknw.build_sknw(ske, iso=False)

    # 4. 规范化图结构 (确保是无向图，且边有权重)
    # sknw 生成的图已经包含 'weight' 属性，代表边的像素长度
    return graph




def visualize_graph(graph, background_image=None):
    """
    graph: sknw 生成的 networkx 图
    background_image: 原始的二值掩膜或骨架图 (可选)
    """
    plt.figure(figsize=(10, 10))
    
    # 1. 如果有背景图，先画背景
    if background_image is not None:
        plt.imshow(background_image, cmap='gray')
    
    # 2. 画边 (Edges) - 用线条连接
    # 遍历每一条边
    for (s, e) in graph.edges():
        # 获取这条边上的所有像素点坐标
        pts = graph[s][e]['pts']
        
        # 注意：pts 是 (row, col) 格式，画图需要 (x, y) 即 (col, row)
        # pts[:, 1] 是 x (列), pts[:, 0] 是 y (行)
        plt.plot(pts[:, 1], pts[:, 0], 'green', linewidth=2, alpha=0.7)

    # 3. 画节点 (Nodes) - 用点标注
    # 收集所有节点的坐标
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    
    #同样注意：ps 是 (row, col)，画图用 (col, row)
    if len(ps) > 0:
        plt.plot(ps[:, 1], ps[:, 0], 'r.', markersize=8)

    plt.title("Road Graph Visualization")
    plt.axis('off') # 关闭坐标轴
    plt.show()


# pred_graph[0][2] # {'pts':array([[ 50, 256], [240, 256], [270, 256], [460, 256]], dtype=int16), 'weight':1.0}

def build_p1_from_polylines(lines):
    p1 = defaultdict(list)

    for line in lines:
        coords = list(line.coords)

        for i in range(len(coords)-1):
            a = coords[i]
            b = coords[i+1]

            p1[a].append(b)
            p1[b].append(a)

    return dict(p1)



def convert_mask_to_graph(mask_path_or_array,save_path=None):
    graph = mask_to_graph(mask_path_or_array)
    lines = []
    for (s, e) in graph.edges():
        pts = graph[s][e]['pts']   # shape = (N, 2), 格式 (y, x)
        line = LineString([(p[1], p[0]) for p in pts])  # 转成 (x, y)
        lines.append(line)

    simplified_lines = [
        line.simplify(1.0, preserve_topology=True)
        for line in lines
    ]

    p = build_p1_from_polylines(simplified_lines)
    
    mask_name = os.path.basename(mask_path_or_array).split('.')[0]
    if save_path is not None:
        with open(os.path.join(save_path, mask_name + '.p'), "wb") as f:
            pickle.dump(p, f)
    
    
if __name__ == "__main__":
    # mask_path = '/home/cz/datasets/roaddataset/chn6/annotations/val'
    # save_path = '/home/cz/codes/segsRoad/topo_metric/target/chn6'
    mask_path = '/home/cz/codes/segsRoad/topo_metric/output/deepglobe/fcn_direction/vis'
    save_path = '/home/cz/codes/segsRoad/topo_metric/output/deepglobe/fcn_direction/graph'
    os.makedirs(save_path,exist_ok=True)
    mask_list = [os.path.join(mask_path, i) for i in os.listdir(mask_path) if (i.endswith('.png') or i.endswith('.jpg'))]
    for mask_path in tqdm(mask_list):
        convert_mask_to_graph(mask_path, save_path)



# gt_path = '/home/cz/datasets/roaddataset/deepglobe/annotations/val/951.png'
# gt_graph = mask_to_graph(gt_path)
# gt_lines = []
# for (s, e) in gt_graph.edges():
#     pts = gt_graph[s][e]['pts']   # shape = (N, 2), 格式 (y, x)
#     line = LineString([(p[1], p[0]) for p in pts])  # 转成 (x, y)
#     gt_lines.append(line)

# simplified_lines = [
#     line.simplify(1.0, preserve_topology=True)
#     for line in gt_lines
# ]

# p = build_p1_from_polylines(simplified_lines)
# with open('./temp1.p', "wb") as f:
#     pickle.dump(p, f)