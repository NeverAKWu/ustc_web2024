from collections import defaultdict

# 输入和输出文件路径
input_file = "../kg_final.txt"
output_file = "../kg_final_opt.txt"


entity_count = defaultdict(int)
relation_count = defaultdict(int)

# 统计出现次数
with open(input_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        head, relation, tail = map(int, line.strip().split('\t'))
        entity_count[head] += 1
        entity_count[tail] += 1
        relation_count[relation] += 1

flag = True
while (flag):
    filtered_lines = []
    flag = False
    for line in lines:
        head, relation, tail = map(int, line.strip().split('\t'))
        
        # 检查条件，过滤满足以下之一的行
        if relation_count[relation] < 50:
            entity_count[head] -= 1
            entity_count[tail] -= 1
            relation_count[relation] -= 1
            flag = True
            continue
        if entity_count[head] < 10 or entity_count[tail] < 10:
            entity_count[head] -= 1
            entity_count[tail] -= 1
            relation_count[relation] -= 1
            flag = True
            continue

        # 如果没有被过滤，添加到新的列表
        filtered_lines.append(line)

    # 将过滤后的列表赋值回原来的变量
    lines = filtered_lines

entity_map = {}
relation_map = {}
next_entity_id = 578  # 实体的映射起始值
next_relation_id = 0  # 关系的映射起始值

with open(output_file, 'w') as f:
    for line in lines:
        head, relation, tail = map(int, line.strip().split('\t'))
            
        # 处理头实体
        if head <= 577:
            new_head = head
        else:
            if head not in entity_map:
                entity_map[head] = next_entity_id
                next_entity_id += 1
            new_head = entity_map[head]
            
        # 处理尾实体
        if tail <= 577:  # 0-577 直接保留
            new_tail = tail
        else:  # 大于等于 578 需要重新映射
            if tail not in entity_map:
                entity_map[tail] = next_entity_id
                next_entity_id += 1
            new_tail = entity_map[tail]
            
        # 处理关系
        if relation not in relation_map:
            relation_map[relation] = next_relation_id
            next_relation_id += 1
        new_relation = relation_map[relation]

        f.write(f"{new_head}\t{new_relation}\t{new_tail}\n")


# # 输出最终结果
# with open(output_file, 'w') as f:
#     for line in lines:
#         f.write(line)
# # 过滤行并写入新文件
# with open(output_file, 'w') as f:
#     for line in lines:
#         head, relation, tail = map(int, line.strip().split('\t'))
#         # 检查条件，过滤满足以下之一的行
#         if relation_count[relation] < 50:
#             entity_count[head] -= 1
#             entity_count[tail] -= 1
#             relation_count[relation] -= 1
#             continue
#         if entity_count[head] < 10 or entity_count[tail] < 10:
#             entity_count[head] -= 1
#             entity_count[tail] -= 1
#             relation_count[relation] -= 1
#             continue
#         # 如果没有被过滤，写入文件
#         f.write(line)

