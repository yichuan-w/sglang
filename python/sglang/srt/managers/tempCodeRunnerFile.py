 # DP_num=2
    # # partation the input_ids into DP_num parts according to the dfs_list and build the DP_num sub trees
    # sub_trees = []
    # for i in range(DP_num):
    #     sub_trees.append(RadixCacheHost())
    # for i,input_ids in enumerate(all_input_ids):
    #     sub_trees[i%DP_num].match_prefix(tuple(input_ids), i)
    # for i in range(DP_num):
    #     print(f"*************** sub tree {i} ****************")
    #     sub_trees[i].pretty_print()
    # for i in range(DP_num):
    #     dfs_list,dfs_request_list,req_num,req_prefill_len_intotal,req_prefill_len_times_reqnum_intotal = sub_trees[i].get_dfs_order_list(sub_trees[i].root_node)
    #     print(f"*************** sub tree {i} ****************")
    #     print(f"req_num {req_num}")
    #     print(f"req_prefill_len_intotal {req_prefill_len_intotal}")
    #     print(f"req_prefill_len_times_reqnum_intotal {req_prefill_len_times_reqnum_intotal}")
    #     print(dfs_list)
    #     print(dfs_request_list)
    
    