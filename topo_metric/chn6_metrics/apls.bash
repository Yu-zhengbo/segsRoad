rm -rf gt_sp.json
rm -rf prop_sp.json

# declare -a arr=( $(jq -r '.test[]' ./data/data_split.json) )
arr=()
for file in ./target/chn6/*.p; do
    if [[ -f "$file" ]]; then
        # 直接使用完整文件名（不带路径）
        filename=$(basename "$file" .p)
        arr+=("$filename")
    fi
done

# source directory
dir=output/chn6/mask2former

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
        python ./chn6_metrics/apls/convert.py "./target/chn6/${gt_graph}" gt_sp.json
        python ./chn6_metrics/apls/convert.py "./${dir}/graph/${i}.p" prop_sp.json
        
        /usr/local/go/bin/go run ./chn6_metrics/apls/main.go gt_sp.json prop_sp.json ./$dir/results/apls/$i.txt  spacenet
    fi
done
python ./chn6_metrics/apls.py --dir $dir