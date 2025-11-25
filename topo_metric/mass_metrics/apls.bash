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
dir=output/mass/diffroad

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
        python ./mass_metrics/apls/convert.py "./target/mass/${gt_graph}" gt_sp.json
        python ./mass_metrics/apls/convert.py "./${dir}/graph/${i}.p" prop_sp.json
        
        /usr/local/go/bin/go run ./mass_metrics/apls/main.go gt_sp.json prop_sp.json ./$dir/results/apls/$i.txt  spacenet
    fi
done
python ./mass_metrics/apls.py --dir $dir