import jieba
import pandas as pd
import ast
import synonyms
import pkuseg
import argparse

def word_split(tags):
    result = []
    for tag in tags :
        seg_list = jieba.cut_for_search(tag)  # jieba分词
        # seg_list = pkuseg.pkuseg().cut(tag) #pkuseg分词
        #去除停用词
        with open('../data/cn_stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
        seg_list = [word for word in seg_list if word not in stopwords]
        with open('../data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
        seg_list = [word for word in seg_list if word not in stopwords]
        for word in seg_list :
            result.append(word)

    # with open('../data/cn_stopwords.txt', 'r', encoding='utf-8') as f:
    #     stopwords = set(line.strip() for line in f)
    # tags = [word for word in tags if word not in stopwords]
    # with open('../data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
    #     stopwords = set(line.strip() for line in f)
    # tags = [word for word in tags if word not in stopwords]
    # for word in tags:
    #     result.append(word)

    #同义词合并
    i = 0
    while(i < len(result)):
        flag = 1
        for j in range(len(result)-1, i, -1):
            if(synonyms.compare(result[i], result[j], seg=False) > 0.7):
                if(len(result[i]) > len(result[j])):
                    del result[j]
                else:
                    del result[i]
                    flag = 0
                    break
        if(flag == 1):
            i = i + 1
    return result
    

def process_data(file_path, output_path, item_type):
    # 读取 CSV 数据
    df = pd.read_csv(file_path)
    data = []
    count = 0

    # 遍历每一行，处理 'Tags' 列
    for index, row in df.iterrows():
        count += 1
        print(f"已完成：{count}")
        # 将 Tags 字符串转换为列表
        s = row['Tags'].strip("{}")
        tags = [tag.strip().strip("'") for tag in s.split(",")]
        split_result = word_split(tags)

        # 根据 item_type 添加数据到列表
        data.append({item_type: row[item_type], 'Tags': split_result})

    # 保存处理结果到 CSV 文件
    pd.DataFrame(data, columns=[item_type, 'Tags']).to_csv(output_path, index=False)

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Index Compressor')
    parser.add_argument('-b', '--books', action='store_true', help='split the books datasets')
    parser.add_argument('-m', '--movies', action='store_true', help='split the movies datasets')
    args = parser.parse_args()

    # 根据命令行参数处理不同的数据
    if args.books:
        process_data('../data/origin/selected_book_top_1200_data_tag.csv', '../data/result/book_test.csv', 'Book')
    elif args.movies:
        process_data('../data/origin/selected_movie_top_1200_data_tag.csv', '../data/result/movie_test.csv', 'Movie')
    else:
        print("Please use '-b' or '-m' to specify the dataset to process.")