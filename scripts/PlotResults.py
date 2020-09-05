import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import importlib
import os
import sys
sys.path.insert(0, '../RISCluster/')

import matplotlib.pyplot as plt

import plotting
importlib.reload(plotting)
import utils

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Enter plotting function and settings."
#     )
#     parser.add_argument(
#         "function",
#         help="Enter plotting function.",
#         choices=["view_detections", "view_cluster_results"]
#     )
#     subparsers = parser.add_subparsers(dest='subcommand')
#
#     # Subparser for view_detections:
#     parser1 = subparsers.add_parser('view_detections')
#     parser1.add_argument(
#         'dataset',
#         help='Enter path to h5 dataset.',
#         default='../../../Data/DetectionData_4s.h5'
#     )
#     parser1.add_argument(
#         'image_index',
#         help='Select 4 indices from dataset to display.'
#         default=''
#     )

    # Subparser for view_cluster_results:



    # ==== Show examples of detections ========================================
    # fname_dataset = '../../../Data/DetectionData_4s.h5'
    # savepath = '../../../Paper/Figures'
    # # [1234, 1003, 1000000, 10000, 888888]
    # image_index = [1234, 1003, 888887, 10000]
    # fig = plotting.view_detections(
    #     fname_dataset,
    #     image_index,
    #     'Detection Examples'
    # )
    # fig.savefig(f'{savepath}/DetectionExamples.png')

# ==== Show examples of clustering performance ============================
# expname = input('Experiment Serial: ')
expname = 'Exp20200904T012346'
exppath = f'../../../Outputs/Trials/{expname}'
runlist = [f for f in os.listdir(f'{exppath}') if "Run" in f]
A = [
        {
            'exppath': f"{exppath}/{f}",
            'show': False,
            'save': True,
            'savepath': f"{exppath}/{f}"
        } for f in os.listdir(f'{exppath}') if "Run" in f]
print(f"Writing {len(A)} images to disk...", flush=True, end="")
# with utils.SuppressStdout():
with ProcessPoolExecutor(max_workers=16) as exec:
    futures = [exec.submit(plotting.view_cluster_results, **a) for a in A]
    for future in as_completed(futures):
        future.result()
print("complete.")
# folder = 'Run_Clusters=5_BatchSz=512_LR=0.001_gamma=0.1_tol=0.001'
# # for folder in runlist:
# path = f"{exppath}/{folder}"
# plotting.view_cluster_results(path, show=False, save=True, savepath=path)
