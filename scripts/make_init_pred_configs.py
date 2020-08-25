import sys
sys.path.insert(0, '../RISCluster/')

from utils import make_pred_configs_batch

loadpath = '../../../Outputs/Models/DCM'
savepath = '../../ConfigFiles'
overwrite = False
make_pred_configs_batch(loadpath, savepath, overwrite)
