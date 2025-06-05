# Copyright (c) OpenMMLab. All rights reserved.
from .basicvsr_net import BasicVSRNet
from .basicvsr_pp import BasicVSRPlusPlus
from .basicvsr_pp_v2 import BasicVSRPlusPlus_PQF,BasicVSRPlusPlus_PQF_v2,BasicVSRPlusPlus_PQF_v2_withB_skip
from .dic_net import DICNet
from .edsr import EDSR
from .edvr_net import EDVRNet
from .glean_styleganv2 import GLEANStyleGANv2
from .iconvsr import IconVSR,IconVSR_restore,IconVSR_restore_wo_refill_mv
from .liif_net import LIIFEDSR, LIIFRDN
from .rdn import RDN
from .real_basicvsr_net import RealBasicVSRNet
from .rrdb_net import RRDBNet
from .sr_resnet import MSRResNet
from .srcnn import SRCNN
from .tdan_net import TDANNet
from .tof import TOFlow
from .ttsr_net import TTSRNet
from .iconvsr_ipb import IconVSR_restore_wo_refill_mv_ipb
from .iconvsr_ipb_par import IconVSR_restore_wo_refill_mv_ipb_fast_domain_dynamic_with_par
from .stdf import STDFNet
from .mfqev2 import MFQEv2
from .dcngan import DCNGAN_Net,discriminator
from .mwgan import DenseMWNet_Mini_PSNR
__all__ = [
    'MSRResNet', 'RRDBNet', 'EDSR', 'EDVRNet', 'TOFlow', 'SRCNN', 'DICNet',
    'BasicVSRNet', 'IconVSR', 'RDN', 'TTSRNet', 'GLEANStyleGANv2', 'TDANNet','IconVSR_restore', 'IconVSR_restore_wo_refill_mv'
    'LIIFEDSR', 'LIIFRDN', 'BasicVSRPlusPlus', 'RealBasicVSRNet', 'BasicVSRPlusPlus_PQF','BasicVSRPlusPlus_PQF_v2','BasicVSRPlusPlus_PQF_v2_withB_skip','IconVSR_restore_wo_refill_mv_ipb',
    'BasicVSRNet', 'IconVSR', 'RDN', 'TTSRNet', 'GLEANStyleGANv2', 'TDANNet','IconVSR_restore','IconVSR_restore_wo_refill','IconVSR_restore_wo_refill_mv',
    'IconVSR_restore_wo_refill_mv_domain','DenseMWNet_Mini_PSNR',
    'LIIFEDSR', 'LIIFRDN', 'BasicVSRPlusPlus', 'RealBasicVSRNet', 'BasicVSRPlusPlus_PQF','BasicVSRPlusPlus_PQF_v2','BasicVSRPlusPlus_PQF_v2_withB_skip','STDFNet','MFQEv2','DCNGAN_Net','discriminator',"IconVSR_restore_wo_refill_mv_ipb_fast_domain_dynamic_with_par"
]
