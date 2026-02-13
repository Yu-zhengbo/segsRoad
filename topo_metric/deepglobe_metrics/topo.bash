# source directory
dir=output/deepglobe/swinlarge

python ./deepglobe_metrics/topo/main.py -savedir $dir
python ./deepglobe_metrics/topo.py -savedir $dir