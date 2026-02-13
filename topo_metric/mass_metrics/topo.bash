# source directory
dir=output/mass/swinlarge

python ./mass_metrics/topo/main.py -savedir $dir
python ./mass_metrics/topo.py -savedir $dir