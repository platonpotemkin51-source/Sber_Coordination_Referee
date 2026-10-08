"""Every turn receives the features of its own run."""
import numpy as np
import pandas as pd
from referee.context import global_context_features


def test_context_respects_run_offsets():
    data={'X':pd.DataFrame({'coverage':[.2,.9]}),'turn_X':np.arange(5,dtype=np.float32).reshape(-1,1),'turn_columns':['position'],'offsets':np.array([0,2,5])}
    out=global_context_features(data,['coverage'])
    np.testing.assert_allclose(out['turn_X'][:,1],[.2,.2,.9,.9,.9])
    np.testing.assert_array_equal(out['turn_X'][:,0],data['turn_X'][:,0])
    assert out['turn_columns']==['position','context_coverage']
