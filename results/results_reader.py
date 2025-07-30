# results数据
import os
import json
import pprint
from typing import List, Union
from collections import defaultdict

from .data_class import Set_Rec, RecordFactory, SettingFactory, Record, Setting

class Results:
    def __init__(self):
        self.root = os.path.join("results", 'records')
        self.remove_illigal_results()
        self.setting_deduplicate()
        self.timestamps = Results.load_timestamps()
        self.Trained_Model_Clean()

    @staticmethod
    def get_setting_statistic(k:str = None, values:List[str]=None):
        '''获取当前所有模型的setting的统计字典
        key对应参数名，默认none，返回所有参数
        value对应参数名出现过的所有参数值
        '''
        if not k:# 返回所有统计量
            settings = Results.load_all_settings()
            return Results.settings_to_setting_statistic(settings)
        else:
            filterd_settings = Results.load_settings_by_keyvalue(k, values)
            return Results.settings_to_setting_statistic(filterd_settings)
    
    @staticmethod
    def set_recs_to_statistic(set_recs:List[Set_Rec]):
        '''计算set_recs的参数统计字典'''
        settings = [x.setting for x in set_recs]
        return Results.settings_to_setting_statistic(settings)
        
    @staticmethod
    def get_setting_statistic_by_dict(d:dict):
        settings = Results.load_all_settings()
        for k, vs in d.items():
            settings = Results.filter_settings_by_keyvalue(settings, k, vs)
        return Results.settings_to_setting_statistic(settings)

    @staticmethod
    def load_rs_by_filter(d: dict)->List[Set_Rec]:
        all_rs = Results.load_all_set_res()
        for k, v in d.items():
            if isinstance(v, list):
                all_rs = Results.filter_key_values(all_rs, k, v)
            else:
                all_rs = Results.filter_keyvalue(all_rs, k, v)
        return all_rs

    @staticmethod
    def filter_key_values(all_sr:List[Set_Rec], k:str, values:List[str]):
        '''根据key和value过滤记录'''
        result = []
        for sr in all_sr:
            for v in values:
                if (k, v) in sr:
                    result.append(sr)
        return result
    
    def filter_settings_by_keyvalue(settings, k:str, values:List[str]):
        filtered_settings = []
        for s in settings:
            if k in s.basic_params and str(s.basic_params[k]) in values:
                filtered_settings.append(s)
                continue
            if k in s.model_params and str(s.model_params[k]) in values:
                filtered_settings.append(s)
        return filtered_settings

    @staticmethod
    def settings_to_setting_statistic(settings, keep_single = True):
        '''生成多个setting的统计字典'''
        set_sta = defaultdict(set)
        for s in settings:
            for k, v in s.basic_params.items():
                v = v if type(v)!= list else str(v)
                set_sta[k].add(str(v))
            for k, v in s.model_params.items():
                v = v if type(v)!= list else str(v)
                set_sta[k].add(str(v))
        if not keep_single:
            set_sta = dict(zip([k for k,v in set_sta.items() if len(v)>1],
                               [v for v in set_sta.values() if len(v)>1]))
        return set_sta

    @staticmethod
    def load_timestamps():
        '''加载records文件夹下的所有记录文件名
        '''
        root = os.path.join("results", 'records')
        #得到文件夹下的所有文件名称
        timestamps = [i.split(".")[0] for i in os.listdir(root)]
        # 将文件名按时间顺序倒序，即较新的文件放在列表前面
        return timestamps[::-1]

    @staticmethod
    def Get_Newest_Record():
        '''获取最新的Record'''
        timestamps = Results.load_timestamps()
        print("newest", timestamps[0])
        return RecordFactory.Get_Record(Results.Load_Record_by_timestamp(timestamps[0]))

    def remove_illigal_results(self):
        '''删除非法记录（record，settings以及trained_model
        删除依据：未达成早停条件的record对应的timestramp对应的result予以删除
        '''
        print("remove_illigal_results...")
        # 读取所有records
        all_records:list[Set_Rec] = self.load_all_set_res()# self.Get_All_Set_Recs()
        # 遍历所有records，找到满足条件的timestamp并删除
        for rec in all_records:
            if not rec.record.is_legal():
                pass
            else:
                print("模型未达成早停条件")
                # Results.Delete_Result(rec.timestamp)

    @staticmethod
    def setting_exists(setting:Setting):
        '''判断是否存在相同的setting文件，存在则返回timestamp，否则返回-1'''
        timestamps = Results.load_timestamps()
        for ts in timestamps:
            ts_set = Results.load_setting(ts)
                
            if Results.load_setting(ts) == setting:
                return ts
        return -1
    
    @staticmethod
    def get_all_records():
        '''获取当前目录下所有record'''
        timestamps = Results.load_timestamps()
        recs = []
        for ts in timestamps:
            recs.append(Results.Load_Record_by_timestamp(ts))
        return recs
    
    def setting_deduplicate(self):
        """
            该函数用于去除重复的Result。
            首先，它会遍历所有的时间戳，并将每个时间戳对应的数据类添加到一个列表中。
            然后，它会创建一个包含所有唯一时间戳的列表。
            如果原始的时间戳列表和唯一时间戳列表的长度不同，说明存在重复的元素。
            在这种情况下，它会遍历原始的时间戳列表，并删除不在唯一时间戳列表中的时间戳。
            最后，它会重新加载数据。
        """
        print("setting_deduplicate...")
        srs:list[Set_Rec] = Results.load_all_set_res()
        
        repeats = defaultdict(list) # 重复记录字典

        for sr in srs:
            repeats[sr.setting].append(sr)
    
        for k, rep_srs in repeats.items():
            if len(rep_srs) > 1:# 重复记录
                print(k, "重复记录", len(rep_srs))
                rep_srs.sort(key=lambda x:list(x.record.get_best_performance().values())[0], 
                         reverse=True) # 正序
                
                for sr in rep_srs[1:]:# 只保留性能最好的
                    print('delete:', sr.timestamp, sr.record.get_best_performance() )
                    Results.Delete_Result(sr.timestamp)

    
    def Trained_Model_Clean(self):
        """删除无效(未训练完成)模型参数
        判断依据：存在trained_model但不存在合法record的文件
        """
        print("trained model clean...")
        model_root = os.path.join("results", 'trained_model')
        model_timestamps = [i.split(".")[0] for i in os.listdir(model_root)]

        for model_ts in model_timestamps:
            if model_ts not in self.timestamps:
                Results.Delete_Trained_Model(model_ts)
    
    @staticmethod
    def Delete_Trained_Model(timestamp:str):
        trained_model_path = Results.TimeStamp_To_Trained_Model_Path(timestamp)
        print("删除模型参数:", trained_model_path)
        Results.delete_file(trained_model_path)

    @staticmethod
    def Delete_Result(timestamp:str):
        '''删除timestamp对应的record,setting以及trained_model'''
        record_path = Results.TimeStamp_To_Record_Path(timestamp)
        setting_path = Results.TimeStamp_To_Setting_Path(timestamp)
        
        Results.delete_file(record_path)
        Results.delete_file(setting_path)
        Results.Delete_Trained_Model(timestamp)
    
    @staticmethod
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
    
    def get_timestamps_of_same_model(self, model_name)->list:
        '''返回同一个模型的所有timestamps'''
        tss = []
        timestamps = Results.load_timestamps()
        for ts in timestamps:
            settings = Results.load_setting_file_by_timestamp(ts)

            if settings and settings['model_name'] == model_name:
                tss.append(ts)
        return tss
    
    @staticmethod
    def filter_keyvalue(rec_sets:List[Set_Rec], k, v)->List[Set_Rec]:
        '''对set_rec进行过滤，
        过滤依据，保留record，training performance最大的rec_set
        '''
        new_rec_sets = []
        for rs in rec_sets:
            if (k, v) in rs:
                new_rec_sets.append(rs)
        return new_rec_sets
    
    @staticmethod
    def filter_dict(rec_sets:List[Set_Rec], d:dict)->List[Set_Rec]:
        """互斥筛选"""
        for k, v in d.items():
            if isinstance(v, list):
                print("list not support")
            else:
                rec_sets = Results.filter_keyvalue(rec_sets, k, v)
        return rec_sets
    
    @staticmethod
    def get_set_rec_by_timestamp(timestamp:Union[list, str])->Union[list, str]:
        '''timestamp不存在任何一个文件（record，setting，trained_model）
        则删除所有对应文件
        '''
        if type(timestamp) == list:
            set_recs = []
            for ts in timestamp:
                set_recs.append(Results.get_set_rec_by_timestamp(ts))
            return set_recs
        set_dict = Results.load_setting_file_by_timestamp(timestamp)
        if set_dict == None:
            print("setting file not found")
            Results.Delete_Result(timestamp)
            return None
        set = SettingFactory.Get_Setting(set_dict)
        rec_dict = Results.load_record_json_by_timestamp(timestamp)
        if rec_dict == None:
            print('rec_dict not find')
            Results.Delete_Result(timestamp)
            return None
        rec = RecordFactory.Get_Record(rec_dict)
        return Set_Rec(timestamp, set, rec)

    @staticmethod
    def load_settings_by_keyvalue(k:str, values:List[str]):
        '''返回满足键值对约束的settings'''
        settings = Results.load_all_settings()
        return Results.filter_settings_by_keyvalue(settings, k, values)
    
    @staticmethod
    def load_all_settings()->List[Setting]:
        '''加载results/settings路径下所有json并初始化对应类'''
        timestamps = Results.load_timestamps()
        # 获取所有setting
        all_setting = []
        for ts in timestamps:
            all_setting.append(Results.load_setting(ts))
        return all_setting

    @staticmethod
    def load_all_set_res()->List[Set_Rec]:
        '''获取所有set_rec数据'''
        all_set_rec = []
        timestamps = Results.load_timestamps()

        for ts in timestamps:
            set_rec = Results.get_set_rec_by_timestamp(ts)
            if set_rec:
                all_set_rec.append(set_rec)
        return all_set_rec
    
    @staticmethod
    def load_rs_model_data(model_name, data_name, best = False):
        '''返回同一个模型的同一个数据集上的rec_set, 
        best:True则返回性能最优的
        False：返回所有
        [注意：适用于查找单个记录，重复查找复杂度过高]
        '''
        filter_d = {"model_name":model_name, 'data_name':data_name}
        all_rec_set = Results.load_all_set_res()
        # 过滤all_rec_set
        rs_of_model_data = Results.filter_dict(all_rec_set, filter_d)

        if len(rs_of_model_data)==0:
            # print("不存在训练数据")
            return None
        if best:
            return max(rs_of_model_data)
        return rs_of_model_data
    
    @staticmethod
    def get_models_of_same_data(data_name):
        '''同一个数据上有训练数据的模型
        '''
        all_rec_set = Results.load_all_set_res()
        rss_of_data = Results.filter_keyvalue(all_rec_set, "data_name", data_name)
        model_names = []
        for rs in rss_of_data:
            model_names.append(rs.setting.model_name)
        return list(set(model_names))


    @staticmethod
    def TimeStamp_To_Record_Path(f_name:str):
        return os.path.join("results","records",f_name+".json")
    
    @staticmethod
    def TimeStamp_To_Setting_Path(f_name:str):
        return os.path.join("results","settings",f_name+".json")
    
    @staticmethod
    def TimeStamp_To_Trained_Model_Path(f_name:str):
        return os.path.join("results","trained_model",f_name+".pt")
    
    @staticmethod
    def load_record_json_by_timestamp(timestamp):
        path = Results.TimeStamp_To_Record_Path(timestamp)
        return Results.__Load_Json(path)

    @staticmethod
    def Load_Record_by_timestamp(timestamp):
        rec_json = Results.load_record_json_by_timestamp(timestamp)
        if rec_json:
            return RecordFactory.Get_Record(rec_json)
        else:
            return None
    
    def Load_Record(self, idx_or_timestamp = 0):
        if type(idx_or_timestamp) == int: # index
            ts = self.timestamps[idx_or_timestamp]
        else:
            ts = idx_or_timestamp
        path = self.TimeStamp_To_Record_Path(ts)

        return self.__Load_Json(path)
    
    def load_setting_file_by_timestamp(timestamp):
        path = Results.TimeStamp_To_Setting_Path(timestamp)
        return Results.__Load_Json(path)
    
    @staticmethod
    def load_setting(timestamp):
        path = Results.TimeStamp_To_Setting_Path(timestamp)
        set_file = Results.__Load_Json(path)
        if not set_file:
            return set_file
        return SettingFactory.Get_Setting(set_file)
    
    @staticmethod
    def __Load_Json(path = None)->dict:
        '''读取指定路径的json文件
        不存在路径则返回None'''
        # print(path)
        if os.path.exists(path):
            with open(path, 'r') as f:
                file_data = json.load(f)
            return file_data
        else:
            return None
