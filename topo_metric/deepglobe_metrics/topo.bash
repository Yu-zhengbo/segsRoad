# source directory
dir=output/deepglobe/diffroad

python ./deepglobe_metrics/topo/main.py -savedir $dir
python ./deepglobe_metrics/topo.py -savedir $dir