from .functions import Losses

from .bpr import BPR
from .lightgcn import LightGCN

<<<<<<< HEAD
from .dns import DNS
from .dnsmn import DNSMN
from .cuco import CuCo
from .ahns import AHNS


from .directau import DirectAU
from .ssm import SSM
from .simgcl import SimGCL
from .xsimgcl import XSimGCL
from .lightgcl import LightGCL

from .dnsgcl import DNSGCL
from .dnsxsimgcl import DNSXSimGCL

from .cpsvd import CPSVD
from .sgformer import SGFormer
from .transgnn import TransGNN
from .gat import GAT
from .sigformer import SIGFormer
from .tag_cf import TAG_CF

'''ours'''
from .cft import CFT
from .cftcl import CFTCL
from .cftcl1 import CFTCL1
=======
'''ours'''
>>>>>>> 792f037 (initial commit)
from .fic import FIC

MODEL_MODEL_MAP = {
                    "bpr": BPR,

<<<<<<< HEAD
                    "dns": DNS,
                    "dnsmn": DNSMN,
                    "cuco": CuCo,
                    "ahns": AHNS,

                    "lightgcn": LightGCN,
                    "directau": DirectAU,
                    "ssm": SSM,
                    "simgcl": SimGCL,
                    "xsimgcl": XSimGCL,
                    "lightgcl": LightGCL,

                    "dnsgcl": DNSGCL,
                    "dnsxsimgcl": DNSXSimGCL,
                    "fic": FIC,
                    "cpsvd": CPSVD,
                    'sgformer': SGFormer,
                    "transgnn": TransGNN,
                    'gat':GAT,
                    "sigformer":SIGFormer,
                    "tag_cf":TAG_CF,
                    "cft":CFT,
                    "cftcl":CFTCL,
                    "cftcl1":CFTCL1
=======

                    "lightgcn": LightGCN,
                    "fic": FIC,
>>>>>>> 792f037 (initial commit)
                    }

ALL_MODELS = list(MODEL_MODEL_MAP.keys())
def show_models():
    print(ALL_MODELS)