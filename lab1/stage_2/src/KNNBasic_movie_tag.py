#使用KKNBasic算法，计算用户-用户相似度，依据相似度训练模型，再用模型对测试集进行预测，并给出平均ndcg
#相似度方法：MSD:0.7964~0.8002；cosine：不行；pearson：0.7964
#数据处理：1.对某些重复打分，时间新的权重高(但是没有多少重复打分)---ndcg变化不明显，甚至下降
#2. 过滤掉评分过少的用户和评分过少的书:划分数据集后再次出现的评分过少怎么解决？copy并连接，翻倍，得到ndcg上升
#3. tag分词，基于tag对书本进行相似度评分，然后根据用户对书本的打分，给出用户对相似书籍的预测给分，给予权重后，加到最终得分
# 计算ndcg仅使用了已打分项目，未打分项目如何处理？抹除未打分项目，选择部分已打分项目作为参考和测试集
#
import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD ,KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split as sup_train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from surprise import accuracy
from sklearn.metrics import ndcg_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# 示例用户-项目评分矩阵
data_test = {
    'User': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'Book': [1, 2, 3, 1, 3, 1, 2, 3, 2, 3],
    'Rate': [5, 4, 3, 4, 2, 3, 5, 4, 2, 1],
    'Time': [
        '2011-03-29T12:48:35+0800',
        '2011-03-29T12:49:35+0800',
        '2011-03-29T12:50:35+0800',
        '2011-03-29T12:51:35+0800',
        '2011-03-29T12:52:35+0800',
        '2011-03-29T12:53:35+0800',
        '2011-03-29T12:54:35+0800',
        '2011-03-29T12:55:35+0800',
        '2011-03-29T12:56:35+0800',
        '2011-03-29T12:57:35+0800'
    ],
    'Tag': ["小说", "爱情", "友情", "小说", "温暖", "", "", "", "", ""]
}
data_test_df = pd.DataFrame(data_test)


data_init = pd.read_csv('../data/movie_score.csv')
#data_init = pd.read_csv('data/movie_score.csv')
#可以更换data来源,test/book/movie
df = data_init[['User', 'Movie', 'Rate','Time','Tag']]
df_copy = df.copy()
#df_temp1 = data_init1[['User', 'Movie', 'Rate','Time']]
print(df_copy)

df_cpoy_book = df.copy()
df_cpoy_book = df_cpoy_book[(df_cpoy_book['Tag'].notna()) & (df_cpoy_book['Tag'] != '')]
# 将浮点数转换为字符串
df_cpoy_book['Tag_temp'] = df_cpoy_book['Tag'].apply(lambda x: str(x) if isinstance(x, (float, int)) else x)

df_cpoy_book['Tag'] = df_cpoy_book['Tag_temp'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

df_cpoy_book['Tag'] = df_cpoy_book['Tag'].fillna('').apply(lambda x: [tag.strip() for tag in x.split(',') if tag.strip()])

# 按 Book 分组，并聚合标签
df_grouped = df_cpoy_book.groupby('Movie').agg({
    'Tag': lambda x: list(set(tag for sublist in x for tag in sublist))  # 合并标签并去重
}).reset_index()

df_cpoy_book = df_grouped
print(df_cpoy_book)

def preprocess_tag(tag_list):
    if not tag_list or all(pd.isna(t) for t in tag_list):
        return []
    return [list(jieba.cut(tag)) for tag in tag_list]

df_cpoy_book['Processed_Tags'] = df_cpoy_book['Tag'].apply(preprocess_tag)

# 假设 df 是你的 DataFrame
tags = df_cpoy_book['Tag'].apply(lambda x: ' '.join(x)).tolist()  # 将每个标签列表转换成字符串

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tags)  # X 是一个稀疏矩阵，每行代表一本书，每列代表一个标签

similarity_matrix1 = cosine_similarity(X)

# 获取所有书本的唯一 ID
unique_books = df_cpoy_book['Movie'].unique()

# 创建从 book_id 到索引的映射
book_to_index = {book_id: idx for idx, book_id in enumerate(unique_books)}

# 反向映射，从索引到 book_id
index_to_book = {idx: book_id for book_id, idx in book_to_index.items()}

missing_books = set(df_cpoy_book['Movie']) - set(book_to_index.keys())
if missing_books:
    print(f"Warning: The following books are missing from the book_to_index mapping: {missing_books}")

# 创建书本-书本-得分的 DataFrame
book_similarity_df = pd.DataFrame(similarity_matrix1, index = unique_books, columns = unique_books)

# 将相似度矩阵展开为长格式
book_similarity_long = book_similarity_df.stack().reset_index()
book_similarity_long.columns = ['Book1', 'Book2', 'Similarity']
book_similarity_long = book_similarity_long[book_similarity_long['Book1'] != book_similarity_long['Book2']]

book_similarity_long[['Book1', 'Book2']] = book_similarity_long.apply(lambda x: sorted([x['Book1'], x['Book2']]), axis=1, result_type='expand')

# 删除重复条目
book_similarity_long = book_similarity_long.drop_duplicates()
print(book_similarity_long)
##得到形如Dataframe的书本-书本相似度


print("Movie tag ok")
df_temp = df_copy

##过滤数据不够的，打分少于五个的用户
user_counts = df_temp.groupby('User')['Rate'].count()
valid_users = user_counts[user_counts >= 2].index
filtered_df = df_temp[df_temp['User'].isin(valid_users)]
print("user filtered")
print(filtered_df)
#过滤掉打分少于两个的
books_counts = filtered_df.groupby('Movie')['Rate'].count()
valid_books = books_counts[books_counts >= 2].index
filtered_df1 = filtered_df[filtered_df['Movie'].isin(valid_books)]
print("book filtered")
print(filtered_df1)


# print(filtered_df1)
#filtered_df1.to_csv('filtered_data.csv', index=False)
df_temp = filtered_df1

# 根据时间设置合并权重
df_temp['Time'] = pd.to_datetime(df_temp['Time'])
# 计算最小时间戳
min_time = df_temp['Time'].min()
# 计算每个时间点与最小时间戳的差值（单位：天）
df_temp['Time_Diff'] = (df_temp['Time'] - min_time).dt.days
# 计算权重，时间越新，权重越高
df_temp['Weight'] = df_temp['Time_Diff'] + 1  # 加1是为了避免权重为0
# 计算加权评分
df_temp['Weighted_Rate'] = df_temp['Rate'] * df_temp['Weight']
# 按用户和书籍分组，计算加权平均评分
df_aggregated = df_temp.groupby(['User', 'Movie']).agg(
    {'Weighted_Rate': 'sum', 'Weight': 'sum','Tag': lambda x: list(x)}
).reset_index()
# 计算最终的加权平均评分
df_aggregated['Weighted_Avg_Rate'] = df_aggregated['Weighted_Rate'] / df_aggregated['Weight']
# 保留原始评分列
df_aggregated['Rate'] = df_aggregated['Weighted_Avg_Rate']
print(" time and rate0 ok?")

# 选择需要的列
df_aggregated = df_aggregated[['User', 'Movie', 'Rate','Tag']]

###############
print(df_aggregated)
print(" time and rate0 ok")



reader = Reader(rating_scale=(0, 5))
# 加载数据
data = Dataset.load_from_df(df_aggregated[['User', 'Movie', 'Rate']], reader)
#数据集划分
trainset, testset = sup_train_test_split(data, test_size=0.2,random_state=42)
filtered_testset = testset
# 使用 KNNBasic 算法
sim_options = {
    'name': 'pearson',  # 分别都使用三种相似度度量
    'user_based': True  # 用户-用户相似度
}
algo = KNNBasic(sim_options=sim_options)
print("create model")
# 训练模型
algo.fit(trainset)
print("training model...")

# 预测测试集
predictions = algo.test(filtered_testset)
# 计算 RMSE
accuracy.rmse(predictions)

# 将 predictions 转换为 DataFrame

predictions_df = pd.DataFrame([
    (pred.uid, pred.iid, pred.r_ui, pred.est) 
    for pred in predictions
], columns=['User', 'Book', 'True_Rate', 'Pred_Rate'])
print("before tag plus")
print(predictions_df)
## 用户-用户相似度
#计算PartA
sorted_predictions1 = predictions_df.sort_values(by=['User', 'Pred_Rate'], ascending=[True, False])
sorted_predictions1.dropna(subset=['True_Rate', 'Pred_Rate'], inplace=True)


def compute_ndcg(group):#当前使用预测得分进行评分，实际可使用顺序号
    true_ratings = group['True_Rate'].tolist()
    pred_ratings = group['Pred_Rate'].tolist()
    return ndcg_score([true_ratings], [pred_ratings], k = 35)

ndcg_scores1 = sorted_predictions1.groupby('User').apply(compute_ndcg)
# 计算平均NDCG
avg_ndcg1 = ndcg_scores1.mean()
# 获取所有用户和所有书本
unique_users = predictions_df['User'].unique()
unique_books = predictions_df['Book'].unique()

# 存储新的预测结果
new_predictions = []
print("ready to list all")

# 遍历所有用户和书本
for user_id in tqdm(unique_users, desc="Processing users"):
    for book_id in unique_books:
        # 获取用户对所有书本的预测打分
        user_ratings = predictions_df[predictions_df['User'] == user_id].set_index('Book')['Pred_Rate']
        
        # 获取书本的相似度
        similar_books = book_similarity_long[book_similarity_long['Book1'] == book_id][['Book2', 'Similarity']].set_index('Book2')
        
        # 只取相似度最高的前十个书本
        if not similar_books.empty:
            top_similar_books = similar_books.nlargest(10, 'Similarity')
            similar_books_with_ratings = top_similar_books.join(user_ratings, how='inner')
            
            if not similar_books_with_ratings.empty:
                weighted_sum = (similar_books_with_ratings['Similarity'] * similar_books_with_ratings['Pred_Rate']).sum()
                total_weight = similar_books_with_ratings['Similarity'].sum()
                
                # 检查 total_weight 是否为零
                if total_weight > 0:
                    new_predicted_rating = weighted_sum / total_weight
                else:
                    new_predicted_rating = user_ratings.mean()
            else:
                new_predicted_rating = user_ratings.mean()
        else:
            new_predicted_rating = user_ratings.mean()
        new_predictions.append({'User': user_id, 'Book': book_id, 'New_Pred_Rate': new_predicted_rating})
# 将新的预测结果转换为 DataFrame
new_predictions_df = pd.DataFrame(new_predictions)
tag_weight = 0.5
'''
new_predictions_df = predictions_df[['User','Book']]
new_predictions_df['New_Pred_Rate'] = predictions_df['True_Rate']*tag_weight
'''
print("use tag to give scores:")
print(new_predictions_df)

#合并得分
final_predictions_df = predictions_df.merge(new_predictions_df, on=['User', 'Book'], how='left')
print(final_predictions_df.columns)
#tag的权重设置为0.5
final_predictions_df['Final_Pred_Rate'] = (final_predictions_df['Pred_Rate'] + 1 * final_predictions_df['New_Pred_Rate'])/(1+tag_weight)

print("final scores after tag plus:")
print(final_predictions_df)

# 按用户分组并排序
sorted_predictions = final_predictions_df.sort_values(by=['User', 'Pred_Rate'], ascending=[True, False])
# 打印排序后的预测结果
print("Sorted predictions by user and predicted rate:")
print(sorted_predictions)

#清除含NAN元素的
sorted_predictions.dropna(subset=['True_Rate', 'Final_Pred_Rate'], inplace=True)
sorted_predictions.rename(columns={'Book': 'Movie'}, inplace=True)
print(sorted_predictions)

## sorted_predictions.to_csv('movie_pred_score.csv', index=False)

def compute_ndcg(group):#当前使用预测得分进行评分，可使用顺序号
    true_ratings = group['True_Rate'].tolist()
    pred_ratings = group['Final_Pred_Rate'].tolist()
    return ndcg_score([true_ratings], [pred_ratings], k = 35)

ndcg_scores = sorted_predictions.groupby('User').apply(compute_ndcg)
# 计算平均NDCG
avg_ndcg = ndcg_scores.mean()
print("before tag ,avg_ndcg = ",avg_ndcg1)
print("after tag ,avg_ndcg = ",avg_ndcg)
