import concurrent.futures
from datetime import datetime
import json
import sys
sys.path.insert(0, '../../RISCluster/')

import h5py
import numpy as np

from RISCluster.processing import processing as process
from RISCluster.utils.utils import notify

msg = 'hello'
subj = 'test'
notify(subj, msg)
