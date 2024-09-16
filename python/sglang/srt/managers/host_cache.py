from typing import Dict, Tuple
from collections import defaultdict


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i
all_input_ids = [
    [128000, 861, 220, 18, 5144, 315, 11495, 22963, 2851, 25, 220],
    [128000, 861, 220, 18, 5144, 315, 11495, 22963, 2851, 25, 220],
    
    [128000, 861, 220, 18, 5144, 315, 11495, 22963, 2851, 25, 220,222],
    
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 256],
    [128000, 861, 220,21, 5144, 315, 11495, 19794, 2851, 25,27],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 220, 17, 13, 220, 18, 13, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 220, 17, 13, 220, 18, 13, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 13, 220, 18, 13, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 13, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 11606, 15784, 40692, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 11606, 15784, 40692, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 11606, 15784, 40692, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 69502, 12301, 47075, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220,21, 5144, 315, 11495, 19794, 2851, 25],
    [128000, 861, 220,21, 5144, 315, 11495, 19794, 2851, 25],
    [128000, 861, 220,21, 5144, 315, 11495, 19794, 2851, 25],
    [128000, 861, 220,21, 5144, 315, 11495, 19794, 2851, 25,26],
    
    [128000, 861, 220,21, 5144, 315, 11495, 19794, 2851, 25,27],
    [128000, 861, 220, 18, 5144, 315, 11495, 22963, 2851, 25, 220,223],
    [128000, 861, 220, 18, 5144, 315, 11495, 22963, 2851, 25, 220,222],
    [128000, 861, 220, 18, 5144, 315, 11495, 22963, 2851, 25, 220,222,222],
    
    ]
all_input_ids=[    [128000, 861, 220, 18, 5144, 315, 11495, 22963, 2851, 25, 220],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 220, 17, 13, 220, 18, 13, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 220, 17, 13, 220, 18, 13, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 13, 220, 18, 13, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 13, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 11606, 15784, 40692, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 11606, 15784, 40692, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 11606, 15784, 40692, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 220, 16, 13, 2009, 68557, 7957, 220, 17, 11606, 14616, 67440, 220, 18, 69502, 12301, 47075, 220, 19, 13, 220, 20, 13, 220, 21, 13, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 256],
    [128000, 861, 220, 21, 5144, 315, 11495, 19794, 2851, 25, 256]]
# import json
# from transformers import AutoTokenizer

# # 加载 Meta-Llama-3-8B-Instruct 的 tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# # 定义处理函数
# def process_prompts_and_store(input_file, all_input_ids):
#     with open(input_file, "r") as infile:
#         for line in infile:
#             # 解析 JSON 行
#             req = json.loads(line.strip())
            
#             # 提取 prompt
#             prompt = req['body']['prompt']
            
#             # 对 prompt 进行编码
#             encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True)
            
#             # 将编码后的结果存储到 all_input_ids 数组中
#             all_input_ids.append(encoded_prompt)

# # 创建一个空的 all_input_ids 数组
# all_input_ids = []

# # 调用函数，处理 /home/ycwang/sglang/examples/usage/arc_requests_output.jsonl 文件
# input_filename = "/home/ycwang/sglang/examples/usage/arc_requests_output.jsonl"
# process_prompts_and_store(input_filename, all_input_ids)

# # 输出 all_input_ids 以检查结果
# print(all_input_ids)  # This will print the list of encoded prompts

class TreeNode:

    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None

        self.worker_ids = set()
        # self.request_id = None
        self.request_num = -1
        self.request_ids=[]
        self.input_len = -1
        self.key = None

import logging
class RadixCacheHost:
    GROUPING_THRESHOLD = 1
   

    def __init__(self, disable=False, grouping_threshold=None):
        self.root_node = TreeNode()
        self.disable = disable
        self.logger = logging.getLogger("RadixTree")
        self.logger.setLevel(logging.INFO) 
        if grouping_threshold is not None:
            self.__class__.GROUPING_THRESHOLD = grouping_threshold

    def reset_worker(self, worker_id):
        queue = [self.root_node]
        while queue:
            cur_node = queue.pop(0)
            for child in cur_node.children:
                if worker_id in child.worker_ids:
                    child.worker_ids.remove(worker_id)
                    if not child.worker_ids:
                        del cur_node.children[child]
                    else:
                        queue.append(child)
                else:
                    continue

    def reset(self):
        self.root_node = TreeNode()

    def match_prefix(self, key, request_id: int) -> Tuple[Dict[int, int], int]:

        def _insert_helper(node, key, request_id, worker_ids=None,len_of_entirekey=-1,num_of_request=-1,child_nodes=None):
            if len(key) == 0:
                return None
            new_node = TreeNode()
            new_node.parent = node
            # new_node.request_id = request_id
            if num_of_request >0:
                new_node.request_ids = request_id
            new_node.worker_ids = worker_ids or set()
            node.children[key] = new_node
            new_node.input_len = len_of_entirekey
            new_node.request_num = num_of_request
            if child_nodes is not None and len(child_nodes) > 0:
                new_node.children = child_nodes
            return new_node

        matched_length = 0
        matched_workers = {}
        matched_req = request_id
        origin_key = key

        node_stack = [(self.root_node, key)]
        while node_stack:
            cur_node, key = node_stack.pop(0)
            for c_key, child in cur_node.children.items():
                prefix_len = match(c_key, key)
                self.logger.debug(f"prefix len {prefix_len} {c_key} {key}")
                if prefix_len == 0:
                    # not matched at all
                    continue
                else:
                    matched_length += prefix_len
                    if matched_length >= len(
                            origin_key) * self.GROUPING_THRESHOLD:
                        self.logger.debug(f"matched length {matched_length} {len(origin_key)}")
                        # merge into the same request affinity group if matched length is more than half
                        self.logger.debug(f"child.request_ids {child.request_ids}")

                        child.request_num += 1
                        child.request_ids.append(request_id)
                        
                    for w in child.worker_ids:
                        matched_workers[w] = matched_length

                    if prefix_len < len(c_key):
                        self.logger.debug(f"split node {c_key} {key}")
                        self.logger.debug(f"prefix len in split {prefix_len} {len(c_key)} {len(key)}")
                        self.logger.debug(f"split node {c_key[:prefix_len]} {c_key[prefix_len:]} {key[prefix_len:]}")
                        
                        
                        

                        # matching complete, split node and insert a new one
                        split_node = _insert_helper(
                            cur_node, c_key[:prefix_len], child.request_ids or [matched_req], child.worker_ids,matched_length,1 if prefix_len == len(key) else 0)
                        
                        _insert_helper(split_node, c_key[prefix_len:],
                                        child.request_ids[:-1] if prefix_len == len(key) else child.request_ids, child.worker_ids,child.input_len,child.request_num -1 if prefix_len == len(key) else child.request_num,child.children)
                        _insert_helper(split_node, key[prefix_len:],
                                       [matched_req],len_of_entirekey=len(origin_key),num_of_request=1)
                        del child.parent.children[c_key]
                    else:
                        self.logger.debug(f"no split node {c_key} {key}")
                        # cur_node.children[c_key].request_id = matched_req
                        if prefix_len < len(key):
                            self.logger.debug(f"keep traversing the tree {child} {key[prefix_len:]}")
                            # keep traversing the tree
                            node_stack.append((child, key[prefix_len:]))
                            break
                        else:
                            # exact match
                            pass
                    return matched_workers, matched_req
            else:
                # no matching, insert a new node and complete
                _insert_helper(cur_node, key, [matched_req],len_of_entirekey=len(origin_key),num_of_request=1)
                return matched_workers, matched_req
            
    def _print_helper(self, node: TreeNode, indent, print_max_len):
        for key, child in node.children.items():
            print(" " * indent, 
                f"{key[:print_max_len]} (Length: {len(key)})",
                f"Request Num: {child.request_num}", 
                f"Request IDs: {child.request_ids}",
                f"Input Len: {child.input_len}", 
                f"Worker IDs: {child.worker_ids}")
            
            

            # if len(child.request_ids) > 0:
            #     for input_ids in child.request_ids:
            #         assert len(all_input_ids[input_ids]) == child.input_len, f"input_ids {input_ids} {all_input_ids[input_ids]} {child.input_len}"
            #         # print('input_len * request_num',child.input_len * child.request_num)
            self._print_helper(child, indent=indent + 2, print_max_len=print_max_len)


    def get_dfs_order_list(self, node: TreeNode):
        dfs_list = []
        dfs_request_list = []
        dfs_prefix_len_list = []
        dfs_prefix_len_times_reqnum_list = []
        stats = {
            'req_num': 0,
            'req_prefill_len_total': 0,
            'req_prefill_len_times_reqnum_total': 0
        }

        def dfs(current_node: TreeNode):
            for key, child in current_node.children.items():
                dfs_list.append(key)
                dfs_request_list.append(child.request_ids)
                stats['req_num'] += child.request_num
                stats['req_prefill_len_total'] += child.input_len * (1 if child.request_num > 0 else 0)
                self.logger.debug(f"input len * request num {child.input_len * child.request_num}")
                stats['req_prefill_len_times_reqnum_total'] += child.input_len * child.request_num
                if child.request_num > 0:
                    dfs_prefix_len_list.append(child.input_len)
                else:
                    dfs_prefix_len_list.append(0)
                dfs_prefix_len_times_reqnum_list.append(child.input_len * child.request_num)
                dfs(child)

        dfs(node)
        return (
            dfs_list,
            dfs_request_list,
            stats['req_num'],
            stats['req_prefill_len_total'],
            stats['req_prefill_len_times_reqnum_total'],
            dfs_prefix_len_list,
            dfs_prefix_len_times_reqnum_list
        )
    def pretty_print(self, print_max_len=30):
        self._print_helper(self.root_node, 0,print_max_len)
        
    def insert(self, key, worker_id: int) -> None:
        '''
            for host cache tree, insertion already happened during scheduling
            therefore, we only need to set the worker_id to traversed nodes
        '''
        node_stack = [(self.root_node, key)]
        while node_stack:
            cur_node, key = node_stack.pop(0)
            for c_key, child in cur_node.children.items():
                prefix_len = match(c_key, key)
                if prefix_len == 0:
                    # not matched at all
                    continue
                # otherwise, must be exact match on nodes traversed
                assert prefix_len == len(c_key)
                # set worker_id and reset the request_id
                cur_node.children[c_key].worker_ids.add(worker_id)
                cur_node.children[c_key].request_id = None
                if prefix_len < len(key):
                    # keep traversing the tree
                    node_stack.append((child, key[prefix_len:]))
                    break
                else:
                    # exact match
                    return

def partition_req_and_construct_trees(dfs_list_new, all_input_ids, DP_num=2):
    partition_size = len(dfs_list_new) // DP_num
    remainder = len(dfs_list_new) % DP_num

    partitions = []
    start = 0

    for i in range(DP_num):
        end = start + partition_size + (1 if i < remainder else 0)
        partition = dfs_list_new[start:end]
        partitions.append(partition)
        # print(f"partition {i}: {partition}")
        start = end

    # Construct trees according to the partitions
    sub_trees = []
    subtree_req_ids = [[] for _ in range(DP_num)]
    for i in range(DP_num):
        # print(f"partitions[i] {partitions[i]}")
        sub_trees.append(RadixCacheHost())

    for i in range(DP_num):
        for j in range(len(partitions[i])):
            # print(f"all_input_ids[partitions[i][j]] {all_input_ids[partitions[i][j]]}")
            sub_trees[i].match_prefix(tuple(all_input_ids[partitions[i][j]]), partitions[i][j])
            subtree_req_ids[i].append(partitions[i][j])
        # print(f"*******!!!!!!!!!!!!!!!!!!!!!!!!!******** sub tree {i} **************!!!!!!!!!!!!!!!!!!!!**")
        # sub_trees[i].pretty_print()

    return sub_trees, subtree_req_ids

def partition_by_prefill_len(dfs_request_list,dfs_prefix_len_list, all_input_ids, dfs_list_new, req_prefill_len_intotal, DP_num=3):
    # Get prefix sum of dfs_prefix_len_list
    prefix_sum = [0]
    print(f"dfs_prefix_len_list {dfs_prefix_len_list}")
    for i in range(len(dfs_prefix_len_list)):
        if dfs_prefix_len_list[i] > 0:
            prefix_sum.append(prefix_sum[-1] + dfs_prefix_len_list[i])
            if len(dfs_request_list[i]) > 0:
                for j in range(len(dfs_request_list[i])-1):
                    prefix_sum.append(prefix_sum[-1])
    prefix_sum = prefix_sum[1:]
    print(prefix_sum)
    assert len(prefix_sum) == len(dfs_list_new)
    tree_prefix_len = req_prefill_len_intotal / DP_num
    print(f"tree_prefix_len {tree_prefix_len}")
    print(f"prefix_sum {req_prefill_len_intotal}")
    
    # Build DP num subtrees
    sub_trees = [RadixCacheHost() for _ in range(DP_num)]
    subtree_req_ids = [[] for _ in range(DP_num)]
    cnt = 0
    
    # Correctly handle the case where prefix sum is less than tree_prefix_len
    for i in range(len(dfs_list_new)):
        print('i len',i,len(dfs_list_new))
        print(f"prefix_sum[i] {dfs_list_new[i]} {tree_prefix_len*cnt}")
        if prefix_sum[i] <= tree_prefix_len * (cnt + 1):
            sub_trees[cnt].match_prefix(tuple(all_input_ids[dfs_list_new[i]]), dfs_list_new[i])
            subtree_req_ids[cnt].append(dfs_list_new[i])
        else:
            cnt += 1
            print('cnt',cnt)
            sub_trees[cnt].match_prefix(tuple(all_input_ids[dfs_list_new[i]]), dfs_list_new[i])
            subtree_req_ids[cnt].append(dfs_list_new[i])
    # Print the subtrees
    for i, tree in enumerate(sub_trees):
        print(f"*******!!!!!!!!!!!!!!!!!!!!!!!!!******** sub tree {i} **************!!!!!!!!!!!!!!!!!!!!**")
        tree.pretty_print()
    
    return sub_trees, subtree_req_ids

if __name__ == "__main__":
    
    host_cache = RadixCacheHost()
    logger = logging.getLogger("Main")
    logger.setLevel(logging.INFO) 

    
    # print len of each input_ids
    for i, input_ids in enumerate(all_input_ids):
        if logger.isEnabledFor(logging.DEBUG):
            print(f"i, len of input_ids {i}, {len(input_ids)}")

    cnt = 0
    for i,input_ids in enumerate(all_input_ids):
        host_cache.match_prefix(tuple(input_ids), cnt)
        cnt += 1
        logger.debug(f"xxxxxxxxxxxxxxxxx input_ids xxxxxxxxxxxxxx {input_ids} {cnt}")
        if logger.isEnabledFor(logging.DEBUG):
            host_cache.pretty_print()
    
    ##TODO: test the matching
    host_cache.pretty_print()
    dfs_list,dfs_request_list,req_num,req_prefill_len_intotal,req_prefill_len_times_reqnum_intotal,dfs_prefix_len_list,dfs_prefix_len_times_reqnum_list = host_cache.get_dfs_order_list(host_cache.root_node)
    assert len(dfs_list) == len(dfs_request_list)
    assert len(all_input_ids) == req_num
    assert req_prefill_len_times_reqnum_intotal == sum([len(input_ids) for input_ids in all_input_ids]),f"req_prefill_len_times_reqnum_intotal {req_prefill_len_times_reqnum_intotal} {sum([len(input_ids) for input_ids in all_input_ids])}"
    print('sum input_id len ',sum([len(input_ids) for input_ids in all_input_ids]))
    print(f"req_num {req_num}")
    print(f"req_prefill_len_intotal {req_prefill_len_intotal}")
    print(f"req_prefill_len_times_reqnum_intotal {req_prefill_len_times_reqnum_intotal}")
    print(dfs_list)
    print(dfs_request_list)
    print(dfs_prefix_len_list)
    print(dfs_prefix_len_times_reqnum_list)
    assert len(dfs_prefix_len_list) == len(dfs_prefix_len_times_reqnum_list)
    
    # first get a list using  dfs_request_list
    dfs_list_new=[]
    for i in range(len(dfs_request_list)):
        if len(dfs_request_list[i]) > 0:
            for j in range(len(dfs_request_list[i])):
                dfs_list_new.append(dfs_request_list[i][j])
    mode_list = ["partition_by_req","round_robin","partation_by_req_prefill_len","partation_by_req_prefill_len_times_reqnum"]
    mode=mode_list[0]
    if mode == "partation_by_req_prefill_len":
        # want to traverse dfs_req_liest and evenly partition the input_ids into DP_num parts, the partation method is accumulate the prefill len and evenly cut off using the total prefill len
        DP_num = 3
        sub_trees, subtree_req_ids = partition_by_prefill_len(dfs_request_list,dfs_prefix_len_list, all_input_ids, dfs_list_new, req_prefill_len_intotal, DP_num)
    elif mode == "partation_by_req_prefill_len_times_reqnum":
        pass

    elif mode == "partition_by_req":
        # first partation dfs_request_list into DP_num parts evenly 
        sub_trees, subtree_req_ids = partition_req_and_construct_trees(dfs_list_new, all_input_ids, 2)
    elif mode=="round_robin":
        DP_num=2
        # partation the input_ids into DP_num parts according to the dfs_list and build the DP_num sub trees
        sub_trees = []
        for i in range(DP_num):
            sub_trees.append(RadixCacheHost())
        for i,input_ids in enumerate(all_input_ids):
            sub_trees[i%DP_num].match_prefix(tuple(input_ids), i)
            print('******** i *********',i)
            sub_trees[i%DP_num].pretty_print()
        for i in range(DP_num):
            print(f"*******!!!!!!!!!!!!!!!!!!!!!!!!!******** sub tree {i} **************!!!!!!!!!!!!!!!!!!!!**")
            sub_trees[i].pretty_print()
        for i in range(DP_num):
            dfs_list,dfs_request_list,req_num,req_prefill_len_intotal,req_prefill_len_times_reqnum_intotal,dfs_prefix_len_list,dfs_prefix_len_times_reqnum_list = sub_trees[i].get_dfs_order_list(sub_trees[i].root_node)
            print(f"*************** sub tree {i} ****************")
            print(f"req_num {req_num}")
            print(f"req_prefill_len_intotal {req_prefill_len_intotal}")
            print(f"req_prefill_len_times_reqnum_intotal {req_prefill_len_times_reqnum_intotal}")
            assert req_prefill_len_times_reqnum_intotal == sum([len(input_ids) for j, input_ids in enumerate(all_input_ids) if j%DP_num == i]),f"req_prefill_len_times_reqnum_intotal {req_prefill_len_times_reqnum_intotal} {sum([len(input_ids) for j, input_ids in enumerate(all_input_ids) if j%DP_num == i])}"
            print(dfs_list)
            print(dfs_request_list)
    
    