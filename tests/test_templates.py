"""Template mining must not learn new templates from validation events."""
import numpy as np
import pandas as pd
from referee.templates import TemplateFeatures


def test_parser_matches_validation_without_creating_clusters():
    data={'documents':['GOAL budget\nhandoff assigning check_budget to ID\ninform acknowledged starting check_budget','GOAL compliance\nhandoff assigning verify_compliance to ID\ninform acknowledged starting verify_compliance'], 'X':pd.DataFrame({'base':[1.,2.]}), 'turn_X':np.zeros((4,1),dtype=np.float32),'turn_columns':['base']}
    parser=TemplateFeatures().fit(data,np.array([0]))
    count=len(parser.miner.drain.clusters)
    transformed=parser.transform(data)
    assert transformed['turn_X'].shape[0]==4
    assert transformed['turn_X'][0,1]==transformed['turn_X'][2,1]
    assert parser.code('system unprecedented semaphore exhaustion')==0
    assert len(parser.miner.drain.clusters)==count
