#!/usr/bin/env python3
"""Independent scalar update oracle for partial shuffle units.
Only tests the CURRENT extension on 33 synthetic rows; no baseline engine runs.
"""
import json, os, subprocess
from pathlib import Path
pg_config=os.environ.get('PG_CONFIG','pg_config')
binpath=Path(subprocess.check_output([pg_config,'--bindir'],text=True).strip())
def sql(text):
    return subprocess.check_output([str(binpath/'psql'),'-X','-A','-t','-v','ON_ERROR_STOP=1','-c',text],text=True).strip()
for mode in ['batch','row','corgipile']:
    options={'profile':'opengauss','max_iterations':5,'batch_size':8,'learning_rate':.05,'decay':1,'tolerance':0,'seed':42,'shuffle':mode}
    if mode=='corgipile':options.update(shuffle_block_rows=7,shuffle_window_blocks=3)
    else:options['shuffle_buffer_size']=64
    spec=json.dumps(options)
    name='partial_oracle_'+mode
    sql(f"SELECT db4ai.drop_model('{name}'); SELECT db4ai.train('{name}','linear_regression','SELECT 1+2*((i-16)::float8/10), (i-16)::float8/10 FROM generate_series(0,32)i','{spec}')")
    actual=json.loads(sql(f"SELECT convert_from(db4ai.export_model('{name}'),'UTF8')"))['weights']
    w=[0.,0.]
    for epoch in range(5):
        order=json.loads(sql(f"SELECT to_json(db4ai.shuffle_indices(33,'{spec}',{epoch}))"))
        assert sorted(order)==list(range(33))
        for off in range(0,len(order),8):
            batch=order[off:off+8];grad=[0.,0.]
            for i in batch:
                x=(i-16)/10.;y=1+2*x;err=w[0]+w[1]*x-y
                grad[0]+=err;grad[1]+=err*x
            for j in range(2):w[j]-=.05*grad[j]/len(batch)
    assert max(abs(a-b) for a,b in zip(actual,w))<1e-13,(mode,actual,w)
    print('PASS independent partial-minibatch reassembly oracle:',mode)
