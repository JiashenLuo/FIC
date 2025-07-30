from . import bpr, dns, fic, lightgcn, directau, ssm, simgcl, dnsxsimgcl, ahns
from . import dnsmn, cuco
from .basic_settings import BasicSettings
from .bpr import BPRSettings

from .dns import DNSSettings
from .dnsmn import DNSMNSettings
from .cuco import CuCoSettings
from .ahns import AHNSSettings

from .lightgcn import LightGCNSettings
from .fic import FICSettings
from .directau import DirectAUSettings
from .ssm import SSMSettings
from .simgcl import SimGCLSettings

from .dnsgcl import DNSGCLSettings
from .dnsxsimgcl import DNSXSimGCLSettings

from .cpsvd import CPSVDSettings
from .sgformer import SGFormerSettings
from .transgnn import TransGNNSettings
from .gat import GATSettings
from .sigformer import SIGFormerSettings
from .tag_cf import TAG_CFSettings
from .cft import CFTSettings
from .xsimgcl import XSimGCLSettings
from .lightgcl import LightGCLSettings
from .cftcl import CFTCLSettings
from .cftcl1 import CFTCL1Settings

Model_PARAMS_TABLE = {"bpr":bpr.model_parameters,
                      
                'dns':dns.model_parameters,
                'dnsmn':dnsmn.model_parameters,
                'cuco':cuco.model_parameters,
                'ahns':ahns.model_parameters,

                'lightgcn':lightgcn.model_parameters,
                'directau':directau.model_parameters,
                'ssm': ssm.model_parameters,
                'simgcl': simgcl.model_parameters,
                "xsimgcl": xsimgcl.model_parameters,
                "lightgcl": lightgcl.model_parameters,
                "gat": gat.model_parameters,
                
                "sgformer": sgformer.model_parameters,
                "transgnn": transgnn.model_parameters,
                "sigformer": sigformer.model_parameters,
                "tag_cf": tag_cf.model_parameters,

                'dnsgcl': dnsgcl.model_parameters,
                'dnsxsimgcl': dnsxsimgcl.model_parameters,
                'fic': fic.model_parameters,

                "cpsvd": cpsvd.model_parameters,
                "cft": cft.model_parameters,
                "cftcl": cftcl.model_parameters,
                "cftcl1": cftcl1.model_parameters}

MODEL_SETTING_MAP = {"bpr": BPRSettings,
                     
                        "dns": DNSSettings,
                        "dnsmn": DNSMNSettings,
                        "cuco": CuCoSettings,
                        "ahns": AHNSSettings,

                        "lightgcn": LightGCNSettings,
                        "directau": DirectAUSettings,
                        "ssm": SSMSettings,
                        "simgcl": SimGCLSettings,
                        'xsimgcl': XSimGCLSettings,
                        "lightgcl": LightGCLSettings,

                        'dnsgcl': DNSGCLSettings,
                        'dnsxsimgcl': DNSXSimGCLSettings,
                        'fic': fic.FICSettings,

                        'cpsvd': CPSVDSettings,
                        "sgformer": SGFormerSettings,
                        "transgnn": TransGNNSettings,
                        "gat": GATSettings,
                        "sigformer":SIGFormerSettings,
                        "tag_cf": TAG_CFSettings,
                        "cft": CFTSettings,
                        "cftcl": CFTCLSettings,
                        "cftcl1": CFTCL1Settings}

from . import fic, bpr, lightgcn
from .basic_settings import BasicSettings
from .bpr import BPRSettings
from .lightgcn import LightGCNSettings
from .fic import FICSettings

Model_PARAMS_TABLE = {

                'fic': fic.model_parameters,

}

MODEL_SETTING_MAP = {
                        'fic': fic.FICSettings,
                        }