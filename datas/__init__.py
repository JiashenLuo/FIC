import os
from .data import DataBasic
from .data_bpr import DataBPR
from .data_direactau import DataDirectAU
from .data_lightgcn import DataLightGCN
from .data_cp import DataCP
from .data_sgformer import DataSGFormer
from .data_sigformer import DataSIGFormer
from .data_cft import DataCFT


DataSSM:DataLightGCN = DataLightGCN
DataSimGCL:DataLightGCN = DataLightGCN
DataDNSGCL:DataLightGCN = DataLightGCN


ALL_DATAS = ['frappe','ml_100k','ml_1m', 'douban_book',"yelp2018","gowalla"]

MODEL_DATA_MAP = {"bpr": DataBPR, 
                  
                  "dns": DataBPR,
                  "dnsmn": DataLightGCN,
                  "cuco": DataLightGCN,
                  "ahns": DataLightGCN,

                    "lightgcn": DataLightGCN,
                    "directau": DataDirectAU,
                    "ssm": DataSSM,
                    "simgcl": DataSimGCL,
                    "xsimgcl": DataSimGCL,
                    "lightgcl": DataBPR,

                    "dnsgcl": DataDNSGCL,
                    "dnsxsimgcl": DataDNSGCL,

                    "cpsvd": DataCP,
                    'sgformer': DataSGFormer,
                    "transgnn": DataLightGCN,
                    'gat': DataBPR,
                    "sigformer": DataSIGFormer,
                    "tag_cf": DataLightGCN,
                    # our models
                    "fic": DataLightGCN,
                    "cft": DataCFT,
                    "cftcl": DataCFT,
                    "cftcl1": DataCFT
                    }

def show_datasets():
    print(ALL_DATAS)