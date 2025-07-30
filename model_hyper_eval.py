from models import *
from trainer import load_model
from results.results_reader import Results
from trainer import ModelDataParameterFactor
from settings import MODEL_SETTING_MAP
from datas import  DataBPR
from results import Results, Set_Rec
from utils.evaluator import Evaluator
from typing import List
from p_view import rss_show

from torch_geometric import seed_everything
seed_everything(2024)
# 模型路径
model_name = "cft"
topk = 20
# ml_100k
ml_100k_paths = {
                "LayerNum 1":"results\\trained_model\Sat Nov  9 15-46-44 2024.pt",
                  "PreferenceSentence 20":"results\\trained_model\Sat Sep  7 20-35-22 2024.pt", 
                  "PreferenceSentence 60":"results\\trained_model\Sat Nov  9 15-46-44 2024.pt",
                "PreferenceSentence 80":"results\\trained_model\Sat Sep  7 20-40-40 2024.pt",
                "None":"results\\trained_model\Sat Nov  9 15-46-44 2024.pt",

                  "LayerNum 2":"results\\trained_model\Sat Nov  9 16-38-54 2024.pt",
                  "LayerNum 3":"results\\trained_model\Sat Nov  9 16-44-40 2024.pt",

                "SPE":"results\\trained_model\Sat Sep  7 20-34-19 2024.pt",
                "SSE":"results\\trained_model\Sat Sep  7 20-27-59 2024.pt",
                "SPE+SSE":"results\\trained_model\Sat Sep  7 20-20-14 2024.pt"
                }
# ml_1m
ml_1m_paths = {
                "LayerNum 1":"results\\trained_model\Sat Sep  7 16-51-01 2024.pt",
                  "PreferenceSentence 1":"results\\trained_model\Wed Oct 23 15-33-53 2024.pt", 
                "PreferenceSentence 20":"results\\trained_model\Sat Sep  7 15-48-47 2024.pt",
                    "PreferenceSentence 80":"results\\trained_model\Sat Sep  7 16-51-01 2024.pt",
                "None":"results\\trained_model\Sat Sep  7 16-51-01 2024.pt",

                   "LayerNum 2":"results\\trained_model\Thu Oct 24 18-00-59 2024.pt",
                "LayerNum 3":"results\\trained_model\Fri Oct 25 10-09-43 2024.pt",

                "SPE":"results\\trained_model\Thu Nov 21 19-40-23 2024.pt",
                "SSE":"results\\trained_model\Sat Sep  7 13-15-00 2024.pt",
                "SPE+SSE":"results\\trained_model\Sat Sep  7 11-02-25 2024.pt"
                }

# douban_book
douban_book_paths = {
                "LayerNum 1":"results\\trained_model\Fri Sep  6 23-08-21 2024.pt",
                  "PreferenceSentence 1":"results\\trained_model\Wed Oct 23 19-44-57 2024.pt", 
                "PreferenceSentence 20":"results\\trained_model\Sat Sep  7 19-44-42 2024.pt",
                    "PreferenceSentence 40":"results\\trained_model\Fri Sep  6 23-08-21 2024.pt",
                "None":"results\\trained_model\Fri Sep  6 23-08-21 2024.pt",

                  "LayerNum 2":"results\\trained_model\Mon Oct 28 20-18-48 2024.pt",
             "LayerNum 3":"results\\trained_model\Mon Oct 28 22-17-15 2024.pt",

                "SPE":"results\\trained_model\Fri Sep  6 22-35-37 2024.pt",
                "SSE":"results\\trained_model\Fri Sep  6 21-43-40 2024.pt",
                "SPE+SSE":"results\\trained_model\Tue Sep  3 04-25-28 2024.pt"
                }

# yelp2018
yelp2018_paths = {
                "LayerNum 1":"results\\trained_model\Fri Nov  8 01-20-47 2024.pt",
                  "PreferenceSentence 5":"results\\trained_model\Wed Nov  6 18-41-44 2024.pt", 
                "PreferenceSentence 9":"results\\trained_model\Thu Nov  7 21-38-24 2024.pt",
                    "PreferenceSentence 16":"results\\trained_model\Fri Nov  8 01-20-47 2024.pt",
                "None":"results\\trained_model\Fri Nov  8 01-20-47 2024.pt",

                 "LayerNum 2":"results\\trained_model\Fri Nov  8 09-12-27 2024.pt",
                   "LayerNum 3":"results\\trained_model\Fri Nov  8 14-49-29 2024.pt",

                "SPE":"results\\trained_model\Sat Nov  9 09-03-05 2024.pt",
                "SSE":"results\\trained_model\Sat Nov  9 06-12-58 2024.pt",
                "SPE+SSE":"results\\trained_model\Sat Nov  9 04-20-24 2024.pt"
                }

# gowalla
gowalla_paths = {
                    "LayerNum 1":"results\\trained_model\Sun Sep  8 20-00-51 2024.pt",
                  "PreferenceSentence 1":"results\\trained_model\Thu Oct 24 11-47-10 2024.pt", 
                "PreferenceSentence 10":"results\\trained_model\Sun Sep  8 16-30-42 2024.pt",
                    "PreferenceSentence 16":"results\\trained_model\Sun Sep  8 20-00-51 2024.pt",
                "None":"results\\trained_model\Sun Sep  8 20-00-51 2024.pt",

                 "LayerNum 2":"results\\trained_model\Fri Oct 25 19-36-10 2024.pt",
                   "LayerNum 3":"results\\trained_model\Sat Oct 26 10-47-50 2024.pt",

                "SPE":"results\\trained_model\Sun Sep  8 15-08-37 2024.pt",
                "SSE":"results\\trained_model\Sun Sep  8 09-40-27 2024.pt",
                "SPE+SSE":"results\\trained_model\Sun Sep  8 00-38-43 2024.pt"
                }

datas = ["ml_100k", "ml_1m", "douban_book", "yelp2018", "gowalla"]
paths = [ml_100k_paths, ml_1m_paths, douban_book_paths, yelp2018_paths, gowalla_paths]

results = []
def load_model_by_path(model_name, data_name):
    params = MODEL_SETTING_MAP[model_name](data_name)
    # setattr(params, "layer_num", 2)
    # setattr(params, "layer_num", 3)
    # setattr(params, "model_arc", "SPE")
    # setattr(params, "model_arc", "SSE")
    setattr(params, "model_arc", "SPE SSE")
    print(params.__dict__)
    model, data = ModelDataParameterFactor.get_model_data_by_params(params)
    return model, data

for data_name, path in zip(datas, paths):

    for i, (k, p) in enumerate(path.items()):
        # if i not in [0, 1, 2, 3, 4]:
        #     continue
        # if i != 5:
        #     continue
        # if i != 6:
        #     continue
        # if i != 7:
        #     continue
        # if i != 8:
        #     continue
        if i != 9:
            continue
        model, data = load_model_by_path(model_name, data_name)

        testlabels = data.Get_Test_Labels()
        uid_rated_items = data.Get_User_Rated_Items()
        try:
            model.load(p)
        except:
            print("load model failed", k, p)
            continue
        model.model_to_cuda()
        # 评估模型效果
        eval_dict = Evaluator.Testing_Evaluate(model.predict(topk, uid_rated_items), testlabels)

        print(model_name, data_name, k, eval_dict)
        results.append((model_name, data_name, k, eval_dict))

    for rs in results:
        print(rs)  
print("=++++++++++++++++++++++++++++++++++++++++++++++++++++++=")
for rs in results:
    print(rs)  