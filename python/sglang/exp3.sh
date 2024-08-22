traverse_list=(1000 1500 2000 2500 3000 3500 4000 4500 5000)
port_list=(50000)
order_options=("prefill-reorder" "decode-reorder" "decode-different-len-reorder" "U-distribution")

for port in "${port_list[@]}"
do 
    # 根据port值设置对应的policy
    if [ "$port" -eq 30000 ]; then
        policy="random"
    elif [ "$port" -eq 50000 ]; then
        policy="reverse_length"
    elif [ "$port" -eq 40000 ]; then
        policy="length"
    else
        policy="unknown"
    fi

    for num_prompts in "${traverse_list[@]}"
    do
        for order in "${order_options[@]}"
        do
            # 构造日志文件名
            file_name="sglang_exp_${num_prompts}_${policy}_${order}.log"
            
            echo "num_prompts: $num_prompts, order: $order" | tee "$file_name"
            
            # 构造order参数
            order_arg="--${order}"
            
            # 将标准输出和标准错误同时重定向到文件和命令行
            python3 bench_serving_new_order.py \
                --backend sglang \
                --dataset-name random \
                --num-prompts "$num_prompts" \
                --port "$port" \
                $order_arg 2>&1 | tee -a "$file_name"
        done
    done
done
