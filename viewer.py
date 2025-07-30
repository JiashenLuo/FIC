
# embedding降维可视化
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import results
from settings import MODEL_SETTING_MAP
from trainer import ModelDataParameterFactor



def embedding_dimension_reduction(embs: np.ndarray)->np.ndarray:
    """embedding降维并归一化"""
    print("T-SNE...")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    embs_tsne = tsne.fit_transform(embs)
    x_min, x_max = np.min(embs_tsne, 0), np.max(embs_tsne, 0)
    embs_tsne = (embs_tsne - x_min) / (x_max - x_min)
    return embs_tsne

# 加载训练好的模型
def load_model(model_filter:dict):
    # 加载参数，模型代码，数据
    best_rs = max(results.Results.load_rs_by_filter(model_filter))
    # 用最优参数初始化setting类
    model_name = best_rs.setting.model_name
    data_name = best_rs.setting.data_name
    settings = MODEL_SETTING_MAP[model_name](data_name)
    params = best_rs.setting.get_params()
    for k, v in params.items():
        setattr(settings, k, v)
    # 初始化并加载最优模型
    model, _ = ModelDataParameterFactor.get_model_data_by_params(settings)
    model.load(best_rs.model_path)
    return model

def diff_model_tsne():
    models = ["LightGCN", "XSimGCL", "FIC"]
    colors = ["red", "blue", "green"]
    datas = ["ml_100k", "ml_1m", "douban_book", "yelp2018", "gowalla"]
    data_names = ["Ml-100k", "Ml-1m", "Douban book", "Yelp2018", "Gowalla"]
    sampling_num = 2000
    filters = []
    for model in models:
        for data in datas:
            filters.append({"model_name": model.lower(), "data_name": data})
    # 画图
    # 设置字体为times new roman
    plt.rcParams['font.family'] = 'Times New Roman'
    # 字号为10
    plt.rcParams['font.size'] = 10
    fig, ax = plt.subplots(len(models), len(datas))
    axes = ax.flatten()
    rand_idx_dict =  {}
    # 物品下采样
    for row in range(len(models)):
        for col in range(len(datas)):
            idx = row*len(datas)+col
            filter = filters[idx]
            print(filter)
            # 获取不同模型的embedding
            try:
                model = load_model(filter)
            except:
                print(f"No model found for {filter}")
            # embedding随机下采样
            user_embs, item_embs = model.encoder_predict()
            
            # 随机下采样
            if datas[col] not in rand_idx_dict:
                sampling_num = min(item_embs.shape[0], sampling_num)
                rand_idx = np.random.choice(item_embs.shape[0], sampling_num, replace=False)
                rand_idx_dict[datas[col]] = rand_idx
            else:
                rand_idx = rand_idx_dict[datas[col]]
            print(rand_idx[:10])
            item_embs = item_embs[rand_idx]
            print(item_embs.shape)
            # embs = torch.nn.Embedding(100, 64)
            embs_tsne = embedding_dimension_reduction(item_embs.data.numpy())
            axes[idx].scatter(embs_tsne[:, 0], embs_tsne[:, 1], s=1,
                              c=colors[row])

            if row == 0:
                axes[idx].set_title(f"{data_names[col]}")
            if col == 0:
                axes[idx].set_ylabel(f"{models[row]}")
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            axes[idx].grid()

    plt.tight_layout()
    plt.show()

def hyper_params_tsne():
    data_names = ["Ml-100k", "Ml-1m", "Douban book", "Yelp2018", "Gowalla"]
    datas = ["ml_100k", "ml_1m", "douban_book", "yelp2018", "gowalla"]
    N_matrix = [[5, 7, 10, 50],
                [5, 10, 15, 15],
                [50, 52, 55, 100] ,
                [50, 50, 50, 50],
                [45, 45, 100, 100]]
    filters = [ ]
    for row, data_name in enumerate(datas):
        for idx in range(4):
            filters.append({"neg_c": N_matrix[row][idx],"model_name": "fic",
                             "data_name": data_name})
    sampling_num = 2000
    # 画图
    # 设置字体为times new roman
    plt.rcParams['font.family'] = 'Times New Roman'
    # 字号为10
    plt.rcParams['font.size'] = 10
    fig, ax = plt.subplots(len(data_names), 4)
    axes = ax.flatten()
    rand_idx_dict =  {}
    for row in range(len(data_names)):
        for col in range(4):
            idx = row*4+col
            # 获取不同模型的embedding
            try:
                filter = filters[idx]
                print(filter)
                model = load_model(filter)
            except:
                print(f"No model found for {filter}")
                continue
            # embedding随机下采样
            user_embs, item_embs = model.encoder_predict()
            if datas[col] not in rand_idx_dict:
                sampling_num = min(item_embs.shape[0], sampling_num)
                rand_idx = np.random.choice(item_embs.shape[0], sampling_num, replace=False)
                rand_idx_dict[datas[col]] = rand_idx
            else:
                rand_idx = rand_idx_dict[datas[col]]
            print(rand_idx[:10])
            item_embs = item_embs[rand_idx]
            print(item_embs.shape)
            
            # embs = torch.nn.Embedding(100, 64)
            embs_tsne = embedding_dimension_reduction(item_embs.data.numpy())
            axes[idx].scatter(embs_tsne[:, 0], embs_tsne[:, 1], s=1)

            axes[idx].set_title(f"neg_c-{N_matrix[row][col]}")
            if col == 0:
                axes[idx].set_ylabel(f"{data_names[row]}")
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            axes[idx].grid()


    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # diff_model_tsne()
    hyper_params_tsne()
    