# source directory
dir=output/chn6/mask2former

python ./chn6_metrics/topo/main.py -savedir $dir
python ./chn6_metrics/topo.py -savedir $dir