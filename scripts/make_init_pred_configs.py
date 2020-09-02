import sys
sys.path.insert(0, '../RISCluster/')

from utils import make_pred_configs_batch

loadpath = '../../../Outputs/Models/DCM/Exp20200823T170830'
savepath = '../../ConfigFiles'
overwrite = False
init_path = make_pred_configs_batch(loadpath, savepath, overwrite)
