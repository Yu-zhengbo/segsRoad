rm -rf gt_mass.json
rm -rf prop_mass.json

# declare -a arr=( $(jq -r '.test[]' ./data/data_split.json) )
arr=()
for file in ./target/mass/*.p; do
    if [[ -f "$file" ]]; then
        # 直接使用完整文件名（不带路径）
        filename=$(basename "$file" .p)
        arr+=("$filename")
    fi
done

# source directory
dir=output/mass/swinlarge

# source directory
# dir=$1

mkdir -p ./$dir/results/apls

echo $dir
# now loop through the above array
for i in "${arr[@]}"   
do
    # gt_graph=${i}__gt_graph_dense_spacenet.p
    echo $i
    gt_graph=${i}.p
    if test -f "./${dir}/graph/${i}.p"; then
        echo "========================$i======================"
        python ./mass_metrics/apls/convert.py "./target/mass/${gt_graph}" gt_mass.json
        python ./mass_metrics/apls/convert.py "./${dir}/graph/${i}.p" prop_mass.json
        
        /usr/local/go/bin/go run ./mass_metrics/apls/main.go gt_mass.json prop_mass.json ./$dir/results/apls/$i.txt  spacenet
    fi
done
python ./mass_metrics/apls.py --dir $dir