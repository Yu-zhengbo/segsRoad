# Road Topology Metrics and Visual Case Selection

本目录用于计算道路分割结果的 TOPO-F1、APLS，并根据逐图指标自动筛选适合可视化展示的样例。当前常用数据集示例为 `chn6`，模型输出目录示例为 `topo_metric/output/chn6/segroadv3`。

## 目录约定

一个模型的评估输出目录建议保持如下结构：

```text
topo_metric/output/chn6/<model_name>/
  vis/                  # 模型预测 mask 或可视化结果，文件名为 <image_id>.jpg/png
  graph/                # convert_mask_to_graph.py 生成的 <image_id>.p
  results/
    topo/               # 每张图的 TOPO 结果 txt
    apls/               # 每张图的 APLS 结果 txt
    apls.json           # APLS 汇总
  topo.json             # TOPO 汇总
```

拓扑可视化图统一放在：

```text
topo_metric/output/chn6/graph_viz/<model_name>/<image_id>.png
topo_metric/output/chn6/graph_viz/target/<image_id>.png
```

其中 `graph_viz/target` 是 GT/reference 拓扑图，自动选图脚本会强制把它放进每张拓扑对比图的第一格。

## Step 1: 模型推理

在仓库根目录运行推理，输出到某个模型目录：

```bash
CUDA_VISIBLE_DEVICES=1 python demo/image_demo_with_inferencer.py \
  <data_path> \
  <config_path> \
  --checkpoint <weight_path> \
  --out-dir topo_metric/output/chn6/<model_name>
```

推理后确认 `topo_metric/output/chn6/<model_name>/vis` 下面有逐图结果。

## Step 2: Mask 转 Graph

编辑 [convert_mask_to_graph.py](convert_mask_to_graph.py) 末尾的路径：

```python
mask_path = './output/chn6/<model_name>/vis'
save_path = './output/chn6/<model_name>/graph'
```

然后在 `topo_metric` 目录下运行：

```bash
cd topo_metric
python convert_mask_to_graph.py
```

输出会写到：

```text
topo_metric/output/chn6/<model_name>/graph/*.p
```

## Step 3: 计算 TOPO

编辑 `topo_metric/chn6_metrics/topo.bash` 里的模型目录：

```bash
dir=output/chn6/<model_name>
```

然后运行：

```bash
cd topo_metric
bash chn6_metrics/topo.bash
```

输出包括：

```text
output/chn6/<model_name>/results/topo/*.txt
output/chn6/<model_name>/topo.json
```

`topo.json` 中 `mean topo` 分别为：

```text
[TOPO-F1, Precision, Recall]
```

## Step 4: 计算 APLS

编辑 `topo_metric/chn6_metrics/apls.bash` 里的模型目录：

```bash
dir=output/chn6/<model_name>
```

然后运行：

```bash
cd topo_metric
bash chn6_metrics/apls.bash
```

输出包括：

```text
output/chn6/<model_name>/results/apls/*.txt
output/chn6/<model_name>/results/apls.json
```

## Step 5: 生成拓扑可视化图

如果已经有 `.p` graph 和对应 RGB 原图，可以用 [visualize_road_graph.py](visualize_road_graph.py) 生成覆盖到卫星图上的拓扑图。

单张图：

```bash
python topo_metric/visualize_road_graph.py \
  --image <rgb_image_path> \
  --graph topo_metric/output/chn6/<model_name>/graph/<image_id>.p \
  --output topo_metric/output/chn6/graph_viz/<model_name>/<image_id>.png
```

批量目录：

```bash
python topo_metric/visualize_road_graph.py --image /data1/datasets/zhengbo/roaddataset/chn6/images/val --graph topo_metric/output/chn6/coanet/graph topo_metric/output/chn6/segroadv3/graph topo_metric/output/chn6/convnext/graph topo_metric/output/chn6/poolformer/graph topo_metric/output/chn6/segformerb5/graph /home/cz/codes/segsRoad/topo_metric/output/chn6/localmamba/graph /home/cz/codes/segsRoad/topo_metric/output/chn6/plainmamba/graph /home/cz/codes/segsRoad/topo_metric/output/chn6/vmamba_40k/graph  topo_metric/output/chn6/swinlarge/graph --output topo_metric/output/chn6/graph_viz --edge-color orange --node-color yellow
```

GT/reference 图建议输出到：

```text
python topo_metric/visualize_road_graph.py --image /data1/datasets/zhengbo/roaddataset/chn6/images/val --graph topo_metric/target/chn6 --output topo_metric/output/chn6/graph_viz/target --edge-color lime --node-color blue
```

## Step 6: 自动选择适合展示的图像

使用 [select_visual_cases.py](select_visual_cases.py) 自动从逐图指标中挑选可视化效果较好的样例。

常用命令，目标模型为 `segroadv3`，并和 `topo_metric/output/chn6` 下所有已有指标的模型比较：

```bash
python topo_metric/select_visual_cases.py \
  --target topo_metric/output/chn6/segroadv3 \
  --compare-all \
  --top-k 30
```

输出目录默认是：

```text
topo_metric/output/chn6/segroadv3/selected_cases/
  selected_cases.csv
  selected_cases.json
  vis/
  graph_compare/
    index.html
    rank_001_<image_id>.png
    rank_002_<image_id>.png
```

直接打开这个文件浏览所有拓扑对比结果：

```text
topo_metric/output/chn6/segroadv3/selected_cases/graph_compare/index.html
```

## 自动选图逻辑

脚本会综合考虑：

- 目标模型自身 APLS 和 TOPO-F1 是否高
- 目标模型相比其它模型是否有明显优势
- 图像是否不是过于简单的 case，使用 TOPO probe 数过滤
- APLS 和 TOPO-F1 是否过度不一致，不一致太大时轻微降权

默认综合质量分：

```text
target_quality = 0.5 * APLS + 0.5 * TOPO-F1
```

最终排序还会加入 margin、复杂度奖励和一致性惩罚。

## 常用选图参数

只和指定模型比较：

```bash
python topo_metric/select_visual_cases.py \
  --target topo_metric/output/chn6/segroadv3 \
  --models topo_metric/output/chn6/swinlarge topo_metric/output/chn6/poolformer \
  --top-k 30
```

只生成 CSV/JSON，不复制 `vis` 图：

```bash
python topo_metric/select_visual_cases.py \
  --target topo_metric/output/chn6/segroadv3 \
  --compare-all \
  --top-k 30 \
  --no-copy-vis
```

只显示参与指标比较的模型拓扑图，但仍然强制包含 `graph_viz/target`：

```bash
python topo_metric/select_visual_cases.py \
  --target topo_metric/output/chn6/segroadv3 \
  --compare-all \
  --top-k 30 \
  --graph-viz-mode compared
```

使用 `graph_viz` 下所有拓扑图，包括没有指标的模型，默认就是这个模式：

```bash
python topo_metric/select_visual_cases.py \
  --target topo_metric/output/chn6/segroadv3 \
  --compare-all \
  --top-k 30 \
  --graph-viz-mode all
```

调整 APLS / TOPO 权重：

```bash
python topo_metric/select_visual_cases.py \
  --target topo_metric/output/chn6/segroadv3 \
  --compare-all \
  --apls-weight 0.4 \
  --topo-weight 0.6 \
  --top-k 30
```

过滤太简单或质量太低的图：

```bash
python topo_metric/select_visual_cases.py \
  --target topo_metric/output/chn6/segroadv3 \
  --compare-all \
  --min-topo-probe-count 10 \
  --min-apls 0.5 \
  --min-topo-f1 0.5 \
  --top-k 30
```

不生成拓扑拼图：

```bash
python topo_metric/select_visual_cases.py \
  --target topo_metric/output/chn6/segroadv3 \
  --compare-all \
  --top-k 30 \
  --no-graph-viz
```

## 结果字段说明

`selected_cases.csv` 和 `selected_cases.json` 中常用字段：

- `rank`: 自动排序后的名次
- `image_id`: 图像 ID
- `rank_score`: 最终排序分
- `target_quality`: 目标模型综合质量分
- `target_apls`: 目标模型 APLS
- `target_topo_f1`: 目标模型逐图 TOPO-F1
- `target_topo_precision`: 目标模型 TOPO precision
- `target_topo_recall`: 目标模型 TOPO recall
- `best_other_model`: 除目标模型外综合质量最高的模型
- `margin_to_best_other`: 目标模型相对最佳其它模型的优势
- `<model_name>_quality`: 某个模型的综合质量分
- `<model_name>_apls`: 某个模型的 APLS
- `<model_name>_topo_f1`: 某个模型的 TOPO-F1
- `graph_compare`: 对应的拓扑对比拼图路径

## 推荐完整命令

如果 `segroadv3` 的 TOPO/APLS 和 `graph_viz` 都已经准备好，下次直接运行：

```bash
python topo_metric/select_visual_cases.py \
  --target topo_metric/output/chn6/segroadv3 \
  --compare-all \
  --top-k 30
```

然后打开：

```text
topo_metric/output/chn6/segroadv3/selected_cases/graph_compare/index.html
```
