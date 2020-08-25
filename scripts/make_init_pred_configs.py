import sys
sys.path.insert(0, '../RISCluster/')

from utils import make_pred_configs

loadpath = '../../../Outputs/Models/DCM'
savepath = '../../ConfigFiles'
overwrite = False
make_pred_configs(loadpath, savepath, overwrite)
