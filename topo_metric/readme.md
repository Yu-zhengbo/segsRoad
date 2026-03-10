# Calculate the metric of road about the TOPO-F1 and APLS

## Step 1 Inference the results of model

````python
CUDA_VISIBLE_DEVICES=1 python demo/image_demo_with_inferencer.py <data_path> <config_path> --checkpoint <weight path> --out-dir <output_path>
````

## Step 2 Convert the mask to graph

````python
python topo_metric/convert_mask_to_graph.py 
# change the mask path and output path in convert_mask_to_graph.py
````

## Step 3 Calculte the TOPO

````python
cd topo_metric
bash deepglobe_metrics/topo.bash
````

## Step 4 Calculte the APLS

````python
bash deepglobe_metrics/apls.bash
````