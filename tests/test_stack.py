"""Candidate rows retain class order and use no labels as features."""
import numpy as np
import pandas as pd
from referee.stack import candidate_features


def test_candidate_features_are_independent_of_targets():
    data={'X':pd.DataFrame({'coverage':[.5]}),'turn_X':np.arange(4,dtype=np.float32).reshape(-1,1),'offsets':np.array([0,4]),'y':np.array([1]),'success':np.array([0]),'fault':np.array([2])}
    p=np.ones((1,7))/7; e=np.full((1,7),.3); candidates=np.array([[-1,0,1,2,3,1,2]])
    before=candidate_features(data,np.array([0]),p,e,candidates)
    data.update(y=np.array([6]),success=np.array([1]),fault=np.array([0]))
    after=candidate_features(data,np.array([0]),p,e,candidates)
    np.testing.assert_array_equal(before,after)
    assert before.shape[0]==7
    np.testing.assert_array_equal(before[:,3:10],np.eye(7))
