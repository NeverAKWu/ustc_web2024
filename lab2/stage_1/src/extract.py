import gzip

# 文件路径
freebase_file = "../data/freebase_douban.gz"
douban_to_fb_file = "../data/douban2fb.txt"
movie_id_map_file = "../data/movie_id_map.txt"
output_file = "../kg_final.txt"

# 读取豆瓣到 Freebase 的映射
def read_douban_to_fb(file):
    douban_to_fb = {}
    with open(file, 'r') as f:
        for line in f:
            douban_id, fb_id = line.strip().split()
            douban_to_fb[douban_id] = fb_id
    return douban_to_fb

# 读取豆瓣到索引的映射
def read_douban_to_index(file):
    douban_to_index = {}
    with open(file, 'r') as f:
        for line in f:
            douban_id, index = line.strip().split()
            douban_to_index[douban_id] = int(index)
    return douban_to_index

# 读取 Freebase 文件并同时提取一跳子图
def extract_one_hop_subgraph(file, fb_entities):
    subgraph = []

    with gzip.open(file, 'rb') as f:
        for line in f:
            line = line.strip()
            triplet = line.decode().split('\t')
            head, relation, tail = triplet[0], triplet[1], triplet[2]
            if head.startswith('<http://rdf.freebase.com/ns/') and tail.startswith('<http://rdf.freebase.com/ns/'):
                # 去掉 URI 前缀并移除尾部 '>'
                head_id = head[len('<http://rdf.freebase.com/ns/'):-1]
                tail_id = tail[len('<http://rdf.freebase.com/ns/'):-1]
                if head_id in fb_entities or tail_id in fb_entities:
                    subgraph.append((head_id, relation, tail_id))
    return subgraph

# 映射到索引
def map_to_indices(subgraph, douban_to_index, fb_to_douban, start_index):
    entity_to_index = {}
    relation_to_index = {}
    current_entity_index = start_index
    current_relation_index = 0

    mapped_triples = []

    for head, relation, tail in subgraph:
        # 映射head
        if head in fb_to_douban:
            head_index = douban_to_index[fb_to_douban[head]]
        else:
            if head not in entity_to_index:
                entity_to_index[head] = current_entity_index
                current_entity_index += 1
            head_index = entity_to_index[head]

        # 映射tail
        if tail in fb_to_douban:
            tail_index = douban_to_index[fb_to_douban[tail]]
        else:
            if tail not in entity_to_index:
                entity_to_index[tail] = current_entity_index
                current_entity_index += 1
            tail_index = entity_to_index[tail]

        # 映射relation
        if relation not in relation_to_index:
            relation_to_index[relation] = current_relation_index
            current_relation_index += 1
        relation_index = relation_to_index[relation]


        mapped_triples.append((head_index, relation_index, tail_index))

    return mapped_triples

# 主函数
def main():
    
    print("Loading mappings...")
    douban_to_fb = read_douban_to_fb(douban_to_fb_file)
    douban_to_index = read_douban_to_index(movie_id_map_file)
    
    fb_entities = set(douban_to_fb.values())
    fb_to_douban = {v: k for k, v in douban_to_fb.items()}

    
    print("Extracting one-hop subgraph...")
    subgraph = extract_one_hop_subgraph(freebase_file, fb_entities)

    
    print("Mapping to indices...")
    mapped_triples = map_to_indices(subgraph, douban_to_index, fb_to_douban, start_index=578)

    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        for head, relation, tail in mapped_triples:
            f.write(f"{head}\t{relation}\t{tail}\n")

    print(f"Completed! Subgraph saved to {output_file}")


if __name__ == "__main__":
    main()