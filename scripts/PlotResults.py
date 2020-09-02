import argparse
from datetime import datetime
import importlib
import os
import sys
sys.path.insert(0, '../RISCluster/')

import matplotlib.pyplot as plt

import plotting
importlib.reload(plotting)

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
    expname = 'Exp20200830T232512'
    exppath = f'../../../Outputs/Trials/{expname}'
    runlist = [f for f in os.listdir(f'{exppath}') if "Run" in f]
    # savepath = '../../../Paper/Figures'
    for path in runlist:
        plotting.view_cluster_results(path, show=True, save=False, savepath=path)
