"""Contract and prediction consistency checks for the submitted entry point."""
import subprocess
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("submission_solution",Path(__file__).resolve().parents[1]/"solution.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
infer=module.infer


def test_platform_reduced_smoke_input(tmp_path):
    test=tmp_path/"test.csv"
    pd.DataFrame({"run_id":["smoke_1","smoke_2"]}).to_csv(test,index=False)
    result=tmp_path/"predictions.csv"
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/"solution.py"),"--train",str(tmp_path/"missing.csv"),"--test",str(test),"--output",str(result)],check=True,cwd=tmp_path)
    prediction=pd.read_csv(result)
    assert list(prediction.columns)==["run_id","label","success","fault_turn"]
    assert prediction.run_id.tolist()==["smoke_1","smoke_2"]
    assert prediction.fault_turn.tolist()==[-1,-1]


def test_selected_fault_turn_matches_selected_class():
    class Runs:
        def predict(self,data,indexes):
            return np.array([[.95,.01,.01,.01,.01,.005,.005],[.01,.98,.002,.002,.002,.002,.002]]),np.array([.8,.2])
    class Turns:
        def predict(self,data,indexes):
            return np.array([[-1,7,8,9,10,11,12],[-1,13,14,15,16,17,18]]),np.array([[.99,.01,.01,.01,.01,.01,.01],[.01,.99,.01,.01,.01,.01,.01]])
    pred=infer({"run_model":Runs(),"turn_model":Turns(),"turn_weight":.5},{"run_id":np.array(["one","two"])})
    assert pred.label.tolist()==["clean","dropped_handoff"]
    assert pred.fault_turn.tolist()==[-1,13]
    assert pred.success.tolist()==[1,0]
