import argparse
from settings import MODEL_SETTING_MAP
import pprint
from itertools import product
import copy
from torch_geometric import seed_everything

from trainer import ModelDataParameterFactor, Trainer
from settings import BasicSettings
from results import Results, Setting, SettingFactory

def dict_type(s):
    '''str(param=value) to dict'''
    s = [x.split("=") for x in s.split(' ')]
    return dict(s)

def generate_param_list(params:BasicSettings):
    search_space = params.Get_Search_Space()
    if not search_space:
        raise NotImplemented("没有设置搜索空间")
    # dict to tuple
    search_tuples = []
    for k, vs in search_space.items():
        search_tuples.append([(k, v) for v in vs])
    # 搜索空间展开
    params_list = [] 
    search_tuples = list(product(*search_tuples))
    for param_tuple in search_tuples:
        temp_params = copy.copy(params)
        for k, v in param_tuple:
            setattr(temp_params, k, v)
        params_list.append(temp_params)

    return params_list, search_tuples

if __name__ == "__main__":
    seed_everything(2024)
    from parse import parse_args
    data_name, model_name, params_dict = parse_args()
    # 获取parameters
    params = MODEL_SETTING_MAP[model_name](data_name)
    if params_dict:# 参数重置
        for k, v in params_dict.items():
            setattr(params, k, v)

    pprint.pprint(params.__dict__, compact=True)
    model, data = ModelDataParameterFactor.get_model_data_by_params(params)
    Trainer(params).Train_BP(model, data)





    