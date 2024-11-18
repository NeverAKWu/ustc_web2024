import jieba
import pandas as pd
import ast
import synonyms
import pkuseg

def word_split(tags):
    result = []
    for tag in tags :
        #seg_list = jieba.cut_for_search(tag)  # jieba分词
        seg_list = pkuseg.pkuseg().cut(tag) #pkuseg分词
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
    


df = pd.read_csv('../data/origin/selected_book_top_1200_data_tag.csv')
book_data = []
count = 0
for index, row in df.iterrows():
    count = count + 1
    print("已完成：", count)
    #把Tags字符串转换为列表
    s = row['Tags'].strip("{}")
    tags = [tag.strip().strip("'") for tag in s.split(",")]
    split_result = word_split(tags)

    book_data.append({'Book': row['Book'], 'Tags': split_result})
pd.DataFrame(book_data, columns = ['Book', 'Tags']).to_csv("../data/result/book_test.csv", index=False)

# df = pd.read_csv('../data/origin/selected_movie_top_1200_data_tag.csv')
# movie_data = []
# count = 0
# for index, row in df.head().iterrows():
#     count = count + 1
#     print("已完成：", count)
#     #把Tags字符串转换为列表
#     s = row['Tags'].strip("{}")
#     tags = [tag.strip().strip("'") for tag in s.split(",")]
#     split_result = word_split(tags)

#     movie_data.append({'Movie': row['Movie'], 'Tags': split_result})
# pd.DataFrame(movie_data, columns = ['Movie', 'Tags']).to_csv("../data/result/movie_test.csv", index=False)