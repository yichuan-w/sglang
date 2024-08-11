# traverse from 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 num-prompts every 500 steps
traverse_list=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000)
# traverse_list=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
for num_prompts in "${traverse_list[@]}"
do
    echo "num_prompts: $num_prompts"
    echo "num_prompts: $num_prompts" >> sglang_by_len_new_order_$num_prompts.log
    python3 /home/ubuntu/my_sglang_dev/sglang/python/sglang/bench_serving_new_order.py  --backend sglang --dataset-name random --num-prompts $num_prompts  --prefill-reorder >> sglang_by_len_new_order_$num_prompts.log
done