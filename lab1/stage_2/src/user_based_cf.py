import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from torch.utils.data import Dataset
import argparse

def create_id_mapping(id_list):
    # 从ID列表中删除重复项并创建一个排序的列表
    unique_ids = sorted(set(id_list))
    
    # 创建将原始ID映射到连续索引的字典
    id_to_idx = {id: idx for idx, id in enumerate(unique_ids, start = 1)}
    
    # 创建将连续索引映射回原始ID的字典
    idx_to_id = {idx: id for id, idx in id_to_idx.items()}
    
    return id_to_idx, idx_to_id

class BookRatingDataset(Dataset):
	def __init__(self, data, user_to_idx, book_to_idx, u_items_list, u_users_list, u_users_items_list, i_users_list):
		self.data = data
		self.user_to_idx = user_to_idx
		self.book_to_idx = book_to_idx
		self.u_items_list = u_items_list
		self.u_users_list = u_users_list
		self.u_users_items_list = u_users_items_list
		self.i_users_list = i_users_list

	def __getitem__(self, index):
		row = self.data.iloc[index]
		user = self.user_to_idx[row['User']]
		book = self.book_to_idx[row['Book']]
		rating = row['Rate'].astype(np.float32)
		u_items = self.u_items_list[user]
		u_users = self.u_users_list[user]
		u_users_items = self.u_users_items_list[user]
		i_users = self.i_users_list[book]

		return (user, book, rating), u_items, u_users, u_users_items, i_users

	def __len__(self):
		return len(self.data)
	
# 按用户分组计算NDCG
def compute_ndcg(group):
    true_ratings = group['Rate'].tolist()
    pred_ratings = group['pred'].tolist()
    return ndcg_score([true_ratings], [pred_ratings], k = 50)

# 去中心化余弦相似度
def cosine_similarity_ignore_missing(ratings_matrix, user1_idx, user2_idx):
    user1_ratings = ratings_matrix[user1_idx]
    user2_ratings = ratings_matrix[user2_idx]
    
    # 找到两个用户都有评分的物品
    common_items = (user1_ratings != -1) & (user2_ratings != -1)
    
    if np.sum(common_items) == 0:  # 如果没有共同评分的物品，返回0
        return 0

    # 计算用户1和用户2的平均评分
    user1_mean = np.mean(user1_ratings[user1_ratings != -1])  # 仅计算有评分的物品
    user2_mean = np.mean(user2_ratings[user2_ratings != -1])  # 仅计算有评分的物品

    # 去均值化评分
    user1_centered = user1_ratings[common_items] - user1_mean
    user2_centered = user2_ratings[common_items] - user2_mean

    # 计算余弦相似度，只考虑共同评分的物品
    dot_product = np.dot(user1_centered, user2_centered)
    norm1 = np.linalg.norm(user1_centered)
    norm2 = np.linalg.norm(user2_centered)
    if norm1 == 0 or norm2 == 0:  # 防止除以零的情况
        return 0
    
    return dot_product / (norm1 * norm2)

# 计算指定用户与其他所有用户的相似度，结果是一个向量
def calculate_user_similarities(ratings_matrix, target_user_idx):
    num_users = ratings_matrix.shape[0]  # 用户的数量
    similarity_vector = np.zeros(num_users)  # 初始化相似度向量

    # 计算目标用户与所有其他用户的相似度
    for i in range(num_users):
        if i != target_user_idx:  # 排除目标用户自己
            similarity = cosine_similarity_ignore_missing(ratings_matrix, target_user_idx, i)
            similarity_vector[i] = similarity  # 存储相似度值

    return similarity_vector

# 计算每个用户之间的相似度，并保存结果
def calculate_all_user_similarities(ratings_matrix):
    num_users = ratings_matrix.shape[0]  # 用户的数量
    similarity_matrix = np.zeros((num_users, num_users))  # 初始化相似度矩阵

    # 遍历每对用户，计算相似度
    for i in tqdm(range(num_users)):
        for j in range(i+1, num_users):  # 只计算上三角部分，避免重复计算
            similarity = cosine_similarity_ignore_missing(ratings_matrix, i, j)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # 对称矩阵

    return similarity_matrix

# 根据相似度预测用户对某个物品的评分
def predict_one_item(ratings_matrix, target_user_idx, target_item_idx, k=5):
    """
    根据用户相似度，预测目标用户对目标物品的评分。
    
    Args:
    - ratings_matrix (DataFrame): 用户-物品评分矩阵。
    - target_user_idx (int): 目标用户的索引。
    - target_item_idx (int): 目标物品的索引。
    - k (int): 用于预测的最大邻居用户数量。
    
    Returns:
    - float: 预测的评分。
    """

    # 获取目标用户的平均评分，忽略未评分的物品
    target_user_ratings = ratings_matrix.iloc[target_user_idx]
    target_user_mean = target_user_ratings[target_user_ratings != -1].mean()

    # 获取目标用户与所有其他用户的相似度
    similarity_vector = calculate_user_similarities(ratings_matrix.to_numpy(), target_user_idx)
    
    # 将目标用户与其他用户的相似度和对应评分结合
    similarities_and_ratings = []
    
    for user_idx, similarity in enumerate(similarity_vector):
        if user_idx == target_user_idx or similarity <= 0:
            continue  # 忽略相似度小于等于0或目标用户自身
        
        user_ratings = ratings_matrix.iloc[user_idx]
        user_rating_for_item = user_ratings[target_item_idx+1]
        
        if user_rating_for_item != -1:  # 忽略未对该物品评分的用户
            # 计算该用户的平均评分
            user_mean = user_ratings[user_ratings != -1].mean()
            # 计算去中心化评分
            centered_rating = user_rating_for_item - user_mean
            similarities_and_ratings.append((similarity, centered_rating))
    
    # 如果没有有效的邻居用户，返回默认评分（比如均值或者-1）
    if not similarities_and_ratings:
        return target_user_mean  # 返回目标用户的均值评分
    
    # 按相似度降序排序，并选取前k个邻居
    similarities_and_ratings.sort(reverse=True, key=lambda x: x[0])
    top_k_neighbors = similarities_and_ratings[:k]
    
    # 计算加权预测评分
    numerator = sum(similarity * rating for similarity, rating in top_k_neighbors)
    denominator = sum(similarity for similarity, _ in top_k_neighbors)
    
    if denominator == 0:
        return target_user_mean  # 防止分母为0，返回目标用户均值评分
    
    predicted_rating = target_user_mean + (numerator / denominator)
    return predicted_rating

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="user-based cf")

    parser.add_argument('-t', '--time', action='store_true', help="use time to modify scores")
    parser.add_argument('-r', '--result', action='store_true', help="store the predict result")
    args = parser.parse_args()

    # 读loaded_data取保存的 CSV 文件
    loaded_data = pd.read_csv('../data/book_score.csv')

    loaded_data['Time'] = pd.to_datetime(loaded_data['Time'])

    # 显示加载的数据
    print(loaded_data)

    user_ids = loaded_data['User'].unique()
    book_ids = loaded_data['Book'].unique()

    user_to_idx, idx_to_user = create_id_mapping(user_ids)
    book_to_idx, idx_to_book = create_id_mapping(book_ids)

    u_items_list, i_users_list = [(0, 0)], [(0, 0)]
    loaded_data['user_map'] = loaded_data['User'].map(user_to_idx)
    loaded_data['book_map'] = loaded_data['Book'].map(book_to_idx)

    # 按用户和书籍分组，并保留每个用户对每本书的最后评分（基于Time列）
    # 先按user_map, book_map和Time列排序，然后去除重复评分，只保留最后的评分
    loaded_data_sorted = loaded_data.sort_values(by=['user_map', 'book_map', 'Time'], ascending=[True, True, False])
    loaded_data_deduped = loaded_data_sorted.drop_duplicates(subset=['user_map', 'book_map'], keep='first')

    if args.time:
        # 尝试根据评分时间对rate进行调整
        max_time = loaded_data_deduped['Time'].max()
        loaded_data_deduped['time_weight'] = loaded_data_deduped['Time'].apply(lambda x: 1 / (1 + (max_time.year - x.year)))
        loaded_data_deduped['Rate'] = loaded_data_deduped['Rate'] * loaded_data_deduped['time_weight']


    # 按映射后的用户 ID 分组
    grouped_user = loaded_data_deduped.groupby('user_map')
    grouped_book = loaded_data_deduped.groupby('book_map')

    # 遍历排序后的分组
    for user, group in tqdm(grouped_user):
        books = group['book_map'].tolist()
        rates = group['Rate'].tolist()
        
        u_items_list.append([(book, rate) for book, rate in zip(books, rates)])

    for book, group in tqdm(grouped_book):
        users = group['user_map'].tolist()
        rates = group['Rate'].tolist()
        
        i_users_list.append([(user, rate) for user, rate in zip(users, rates)])

    # 训练集和测试集划分
    train_data, test_data = train_test_split(loaded_data_deduped, test_size=0.2, random_state=42)

    # 创建训练集评分矩阵,也就是用户-物品评分矩阵，行是用户，列是书籍，值是评分
    train_matrix = train_data.pivot(index='user_map', columns='book_map', values='Rate')

    # 填充NaN值（表示用户未评分的项）为-1
    train_matrix = train_matrix.fillna(-1)


    # 预测测试集评分
    tqdm.pandas(desc="Predicting Ratings")
    test_data['pred'] = test_data.progress_apply(lambda row: predict_one_item(train_matrix, row['user_map']-1, row['book_map']-1, k=5), axis=1)

    # 按用户分组计算NDCG
    grouped_test = test_data.groupby('user_map')
    ndcg_per_user = []
    all_groups = []
    for i, group in tqdm(grouped_test):
        try:
            ndcg_per_user.append(compute_ndcg(group))

            group_selected = group[['user_map', 'book_map', 'Rate', 'pred']]
            all_groups.append(group_selected)
        except:
            continue

    # ndcg_per_user = grouped_test.progress_apply(lambda group: compute_ndcg(group))
    if args.result:
        # 合并所有的分组数据
        all_groups_df = pd.concat(all_groups, ignore_index=True)
        # 将汇总数据写入一个 CSV 文件
        all_groups_df.to_csv('../data/predict_result.csv', index=False)

    # 平均NDCG
    mean_ndcg = np.mean(ndcg_per_user)
    print(f"Mean NDCG: {mean_ndcg}")



# # 创建用户-物品评分矩阵，行是用户，列是书籍，值是评分
# user_item_matrix = loaded_data_deduped.pivot(index='user_map', columns='book_map', values='Rate')

# # 填充NaN值（表示用户未评分的项）为-1
# user_item_matrix = user_item_matrix.fillna(-1)

# # 打印用户-物品评分矩阵
# print(user_item_matrix)
# print(predict_one_item(user_item_matrix, 2, 0, 5))
# print(cosine_similarity_ignore_missing(user_item_matrix.values, 0, 1))
# print(calculate_user_similarities(user_item_matrix.values, 0))

# # 计算所有用户之间的相似度
# similarity_matrix = calculate_all_user_similarities(user_item_matrix.values)

# # 将相似度矩阵转换为DataFrame并确保索引从1开始
# similarity_df = pd.DataFrame(similarity_matrix, index=range(1, len(user_item_matrix) + 1), columns=range(1, len(user_item_matrix) + 1))

# # 保存为CSV文件
# similarity_df.to_csv('user_similarities.csv')






