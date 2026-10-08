"""Train-only Drain templates and observable event transitions."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from referee.events import SNAKE


def event_lines(document):
    return document.splitlines()[1:]


def template_input(line):
    line=SNAKE.sub(' SYMBOL ',line)
    return re.sub(r'\s+',' ',re.sub(r'[^a-zA-Z<>\s]',' ',line)).strip()


class TemplateFeatures:
    def fit(self, data, training):
        config=TemplateMinerConfig()
        config.drain_sim_th=.65
        config.drain_depth=4
        config.profiling_enabled=False
        self.miner=TemplateMiner(config=config)
        counts=Counter(template_input(line) for i in training for line in event_lines(data['documents'][i]))
        for text in sorted(counts):
            self.miner.add_log_message(text)
        self.lookup={}
        self.frequency=Counter()
        for text,count in counts.items():
            cluster=self.miner.match(text,full_search_strategy='fallback')
            code=cluster.cluster_id if cluster is not None else 0
            self.lookup[text]=code
            self.frequency[code]+=count
        self.top_codes=[c for c,_ in self.frequency.most_common(160) if c]
        print(f'Drain learned {len(self.miner.drain.clusters)} templates; {len(counts)} unique training messages',flush=True)
        return self

    def code(self, line):
        text=template_input(line)
        if text in self.lookup:
            return self.lookup[text]
        cluster=self.miner.match(text,full_search_strategy='fallback')
        code=cluster.cluster_id if cluster is not None else 0
        self.lookup[text]=code
        return code

    def transform(self, data):
        run_rows,turn_rows=[],[]
        for document in data['documents']:
            lines=event_lines(document)
            codes=[self.code(line) for line in lines]
            counts=Counter(codes); n=len(codes)
            f={}
            for c in self.top_codes:
                f[f'template_count_{c}']=counts[c]
                f[f'template_share_{c}']=counts[c]/max(1,n)
            sequence=np.zeros(64,dtype=np.float32)
            grouped=np.zeros(64,dtype=np.float32)
            latest_by_task={}
            tasks=[]
            for line in lines:
                tokens=SNAKE.findall(line)
                tasks.append(tokens[0] if tokens else '')
            by_task=defaultdict(list)
            for i,(code,task) in enumerate(zip(codes,tasks)):
                if i:
                    sequence[(codes[i-1]*1009+code*9176)%64]+=1
                if task:
                    if task in latest_by_task:
                        grouped[(latest_by_task[task]*1009+code*9176)%64]+=1
                    latest_by_task[task]=code
                    by_task[task].append(i)
            for k in range(64):
                f[f'template_transition_{k}']=sequence[k]/max(1,n)
                f[f'template_task_transition_{k}']=grouped[k]/max(1,n)
            f['template_unknown_share']=counts[0]/max(1,n)
            f['template_unique']=len(counts)
            f['template_max_repetition']=max(counts.values(),default=0)
            local=[]
            for i,code in enumerate(codes):
                indexes=by_task[tasks[i]] if tasks[i] else []
                before=[j for j in indexes if j<i]; after=[j for j in indexes if j>i]
                local.append([code,codes[i-1] if i else 0,codes[i+1] if i+1<n else 0,codes[before[-1]] if before else 0,codes[after[0]] if after else 0,counts[code],sum(codes[j]==code for j in indexes),float(i==codes.index(code)),self.frequency[code]])
            turn_rows.append(np.array(local,dtype=np.float32))
            run_rows.append(f)
        added=np.concatenate(turn_rows)
        if len(added)!=len(data['turn_X']):
            raise ValueError('Template and event turns do not align')
        out=dict(data)
        out['X']=pd.concat([data['X'],pd.DataFrame(run_rows).astype(np.float32)],axis=1)
        out['turn_X']=np.hstack([data['turn_X'],added])
        out['turn_columns']=data['turn_columns']+['template_id_current','template_id_previous','template_id_next','template_id_task_previous','template_id_task_next','template_count','template_task_count','template_first','template_training_frequency']
        return out
