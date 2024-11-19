# ustc_web2024

ustc_web2024 by 1303

## Lab1

### Stage1

cn_stopwords.txt和hit_stopwords.txt为停用词表
word_split.py为分词代码

**运行方法：**

1. 切换目录至`ustc_web2024/lab1/stage_1/src`
2. 建立倒排索引表`python3 ./inverted_index.py [-h] [--input_file INPUT_FILE] [-b] [-m]`
3. 压缩索引`python3 ./index_compress.py [-h] [--index_file INDEX_FILE] [-b] [-m]`
4. 索引解压（可选，仅用作测试）`python3 ./index_decompress.py [-h] [--index_file INDEX_FILE] [-b] [-m]`
5. 执行布尔查询`python3 ./Boolean_query.py [-h] [--index_file INDEX_FILE] [-d] [-v] [-b] [-m]`

说明：

- 默认（建表/压缩/查询）类型为书籍
- 请使用`-h`参数查看详细使用说明
