#使用KKNBasic算法，计算用户-用户相似度，依据相似度训练模型，再用模型对测试集进行预测，并给出平均ndcg
#相似度方法：MSD:0.7964~0.8002；cosine：；pearson：
#情景处理：1.对某些重复打分，时间新的权重高(但是没有多少重复打分)---ndcg变化不明显，甚至下降
#2. 过滤掉评分过少的用户和评分过少的书:划分数据集后再次出现的评分过少怎么解决？copy并连接，翻倍，得到ndcg上升
#计算ndcg仅使用了已打分项目，未打分项目如何处理？抹除未打分项目，选择部分已打分项目作为参考和测试集
#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD ,KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split as sup_train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from surprise import accuracy
from sklearn.metrics import ndcg_score
from collections import defaultdict
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
# 示例用户-项目评分矩阵
data_test = {
    'User': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'Book': [1, 2, 3, 1, 3, 1, 2, 3, 2, 3],
    'rate': [5, 4, 3, 4, 2, 3, 5, 4, 2, 1]
}

data_init = pd.read_csv('data/book_score.csv')
#data_init1 = pd.read_csv('data/movie_score.csv')
df = data_init[['User', 'Book', 'Rate','Time','Tag']]
df_copy = df.copy()
#df_temp1 = data_init1[['User', 'Movie', 'Rate','Time']]
print(df_copy)

# 过滤掉打分为0的部分
#df_temp = df_copy[df_copy['Rate'] != 0]
df_temp = df_copy
# 统计每个用户的打分次数
user_counts = df_temp['User'].value_counts()
# 统计每个项目的评分次数
item_counts = df_temp['Book'].value_counts()
# 过滤掉打分少于 10 次的用户
filtered_data_by_user = df_temp[df_temp['User'].map(user_counts) >= 10]
# 过滤掉已有评分少于2次的项目
filtered_data = filtered_data_by_user[filtered_data_by_user['Book'].map(item_counts) >= 2]
print(filtered_data)
filtered_data.to_csv('filtered_data.csv', index=False)
df_temp = filtered_data

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
df_aggregated = df_temp.groupby(['User', 'Book']).agg(
    {'Weighted_Rate': 'sum', 'Weight': 'sum'}
).reset_index()

# 计算最终的加权平均评分
df_aggregated['Weighted_Avg_Rate'] = df_aggregated['Weighted_Rate'] / df_aggregated['Weight']

# 保留原始评分列
df_aggregated['Rate'] = df_aggregated['Weighted_Avg_Rate']

# 选择需要的列
df_aggregated = df_aggregated[['User', 'Book', 'Rate']]

###############

#df_aggregated = df_temp.groupby(['User', 'Book']).agg({'Rate': 'mean'}).reset_index()
#df_aggregated1 = df_temp1.groupby(['User', 'Movie']).agg({'Rate': 'mean'}).reset_index()
print(df_aggregated)

##user_book_matrix = df_temp.pivot(index='User', columns='Book', values='Rate')

##user_book_matrix = user_book_matrix.fillna(0)
##print(user_book_matrix)

#协同过滤
#user_similarity = cosine_similarity(user_book_matrix)

# 定义函数来预测用户对未评分项目的评分

'''
#SVD
df = pd.DataFrame(data_test)

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 创建 Surprise 数据集
reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(train_df[['User', 'Book', 'rate']], reader)
test_data = Dataset.load_from_df(test_df[['User', 'Book', 'rate']], reader)

# 初始化 SVD 模型
svd = SVD()
algo = KNNBasic()

# 训练模型
trainset = train_data.build_full_trainset()
svd.fit(trainset)

# 获取所有用户和项目
all_users = df['User'].unique()
all_items = df['Book'].unique()

# 预测用户对所有项目的评分
predictions = []
for user in all_users:
    for item in all_items:
        pred = svd.predict(user, item)
        predictions.append((user, item, pred.est))

# 将预测结果转换为 DataFrame
predictions_df = pd.DataFrame(predictions, columns=['User', 'Book', 'predicted_rate'])

# 合并实际评分和预测评分
merged_df = pd.merge(predictions_df, df, on=['User', 'Book'], how='left')
merged_df.rename(columns={'Rate': 'Actual_Rate'}, inplace=True)

#print(predictions_df)
# 过滤掉已评分的项目
#known_rates = set(zip(df['User'], df['Book']))
#predictions_df = predictions_df[~predictions_df[['User', 'Book']].apply(tuple, axis=1).isin(known_rates)]

# 对每个用户按预测评分排序
sorted_predictions = merged_df.sort_values(by=['User', 'predicted_rate'], ascending=[True, False])

# 打印每个用户的前 N 个推荐项目
N = 5
for user in all_users:
    user_recommendations = sorted_predictions[sorted_predictions['User'] == user].head(N)
    ##print(user_recommendations)
    print(f"User {user} recommendations:")
    print(user_recommendations[['Book', 'predicted_rate','rate']])
    print()
'''

reader = Reader(rating_scale=(0, 5))

# 加载数据
data = Dataset.load_from_df(df_aggregated[['User', 'Book', 'Rate']], reader)
#data1 = Dataset.load_from_df(df_aggregated1[['User', 'Movie', 'Rate']], reader)



trainset, testset = sup_train_test_split(data, test_size=0.2,random_state=42)


#testset_df = pd.DataFrame(testset,columns=['User', 'Book', 'Rate'])

##test_item_counts = testset_df.groupby('Book')['Rate'].count()
##valid_books = test_item_counts[test_item_counts >= 2].index

##filtered_testset_df = testset_df[testset_df['Book'].isin(valid_books)]
'''filtered_testset_df = pd.concat([testset_df,testset_df.copy()],ignore_index = True)
print(filtered_testset_df)

filtered_testset = [(uid, iid, r_ui) for uid, iid, r_ui in filtered_testset_df.itertuples(index=False)]
'''
filtered_testset = testset
# 使用 KNNBasic 算法
sim_options = {
    'name': 'pearson',  # 分别都使用三种相似度度量
    'user_based': True  # 用户-用户相似度
}
algo = KNNBasic(sim_options=sim_options)
print("create")
# 训练模型
algo.fit(trainset)
print("fitting")

# 预测测试集
predictions = algo.test(filtered_testset)
## for pred in predictions:
##    print(f"User {pred.uid} - Book {pred.iid}: Actual Rate {pred.r_ui}, Predicted Rate {pred.est}")

# 计算 RMSE
accuracy.rmse(predictions)

# 将 predictions 转换为 DataFrame

predictions_df = pd.DataFrame([
    (pred.uid, pred.iid, pred.r_ui, pred.est) 
    for pred in predictions
], columns=['User', 'Item', 'True_Rate', 'Pred_Rate'])

'''# 定义函数来获取排序序号
def get_rankings(df):
    actual_ranks = df['True_Rate'].rank(method='first', ascending=False).astype(int) - 1
    predicted_ranks = df['Pred_Rate'].rank(method='first', ascending=False).astype(int) - 1
    return actual_ranks, predicted_ranks

# 对每个用户进行排序
user_grouped = predictions_df.groupby('User').apply(get_rankings)
user_grouped = user_grouped.apply(pd.Series, index=['True_Rank', 'Pred_Rank']).reset_index()

# 合并排序结果
predictions_df = predictions_df.merge(user_grouped, on='User')

print(predictions_df)
'''
# 按用户分组并排序
sorted_predictions = predictions_df.sort_values(by=['User', 'Pred_Rate'], ascending=[True, False])
#sorted_predictions = predictions_df
# 打印排序后的预测结果
print("Sorted predictions by user and predicted rate:")
print(sorted_predictions)

def compute_ndcg(group):#当前使用预测得分进行评分，实际可使用顺序号
    true_ratings = group['True_Rate'].tolist()
    pred_ratings = group['Pred_Rate'].tolist()
    return ndcg_score([true_ratings], [pred_ratings], k = 35)

ndcg_scores = sorted_predictions.groupby('User').apply(compute_ndcg)
# 计算平均NDCG
avg_ndcg = ndcg_scores.mean()
print("avg_ndcg = ",avg_ndcg)
