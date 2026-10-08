"""Training normalization and padded sequence correctness."""
import numpy as np
import pandas as pd
import torch
from referee.sequence import normalize_training,SequenceNetwork,batch_arrays


def test_validation_outliers_do_not_change_training_normalization():
    data={'X':pd.DataFrame({'x':[1.,2.,1000.]}),'turn_X':np.array([[1.],[2.],[1000.]],dtype=np.float32),'offsets':np.arange(4)}
    _,_,stats=normalize_training(data,np.array([0,1]))
    data['X'].loc[2,'x']=1e9; data['turn_X'][2,0]=1e9
    _,_,other=normalize_training(data,np.array([0,1]))
    for key in stats:
        np.testing.assert_array_equal(stats[key],other[key])


def test_onset_head_masks_padding_for_short_runs():
    network=SequenceNetwork(2,3,24).eval()
    run=np.zeros((2,2),np.float32); turns=np.zeros((6,3),np.float32); offsets=np.array([0,2,6])
    with torch.inference_mode():
        labels,success,onset=network(*batch_arrays(run,turns,offsets,np.array([0,1])))
    assert labels.shape==(2,7)
    assert success.shape==(2,)
    assert torch.all(onset[0,2:]==-1e4)
