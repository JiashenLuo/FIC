"""
本脚本评估已经训练好的模型的recall以及dncg值。
"""
# 导入必要的库

from models import *
from trainer import load_model
from results.results_reader import Results
from datas import  DataBPR
from utils.evaluator import Evaluator

results:list = [] # [[model_name, data_name, eval_dict]]
models = ["dnsgcl"]
datas = ["ml_100k", "ml_1m", "douban_book", "yelp2018", "gowalla"]


# 设置必要参数
topk = 20
for data_name in datas:
    # 导入测试数据
    databasic = DataBPR(data_name)
    testlabels = databasic.Get_Test_Labels()
    uid_rated_items = databasic.Get_User_Rated_Items()

    for model_name in [i.lower() for i in models]:

        # 导入已完成训练的模型
        print(f"\n[{model_name}, {data_name}]\n",)
        try:
            model = load_model(model_name, data_name)
        except:
            print(f"模型{model_name}在{data_name}不存在已保存的模型")
            continue
        print(model.params.__dict__)
        # model.load(model_data_sr.model_path)
        model.model_to_cuda()
        # 评估模型效果
        eval_dict = Evaluator.Testing_Evaluate(model.predict(topk, uid_rated_items), testlabels)

        print(model_name, data_name, eval_dict)
        results.append([model_name, data_name, eval_dict])



    for rs in results:
        print(rs)
        