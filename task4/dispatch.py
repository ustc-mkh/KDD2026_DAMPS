"""Conservative, resumable background queue; never controls other users' jobs."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--gpus', default='4,5')
a = p.parse_args()
root = Path(a.root).resolve()
repo = Path(__file__).resolve().parents[1]
lock = (root / 'queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
for k in list(os.environ):
    if k.lower().endswith('_proxy') or k == 'SSH_AUTH_SOCK': os.environ.pop(k, None)
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  PYTHONHASHSEED='999', MPLBACKEND='Agg', MPLCONFIGDIR=str(root/'tmp/matplotlib'),
                  NO_PROXY='*', no_proxy='*')
plan = json.loads((repo/'task4/plan.json').read_text())
running = {}
failed = {p.parent.name for p in (root/'runs').glob('*/failure.json')}

def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return {x[0]: x[1:] for line in raw.splitlines() if (x := [s.strip() for s in line.split(',')])}

def resources_ok():
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line and line.split()[1].isdigit()}
    return os.getloadavg()[0] < os.cpu_count()*0.5 and mem['MemAvailable'] > 64*1024**2 and shutil.disk_usage(root).free > 100*1024**3

def run_id(job): return f"{job['dataset']}__{job['variant']}__seed{job['seed']}"

def ready(job):
    d = root/'data'/job['dataset']
    return all((d/f).exists() for f in [job['dataset']+'.inter', 'image_feat.npy', 'text_feat.npy'])

while True:
    for gpu, (proc, log, job) in list(running.items()):
        ret = proc.poll()
        if ret is not None:
            log.close()
            if ret != 0:
                failed.add(run_id(job))
                (root/'runs'/run_id(job)/'failure.json').write_text(json.dumps({'returncode':ret}))
            del running[gpu]
    complete = [run_id(j) for j in plan if (root/'runs'/run_id(j)/'result.json').exists()]
    pending = [j for j in plan if run_id(j) not in complete and run_id(j) not in failed and j not in [v[2] for v in running.values()]]
    status = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  completed=complete, failed=sorted(failed), running={k:dict(v[2],pid=v[0].pid) for k,v in running.items()},
                  waiting_data=[run_id(j) for j in pending if not ready(j)], pending=[run_id(j) for j in pending])
    temporary=root/'queue-status.json.tmp'
    temporary.write_text(json.dumps(status,indent=2)); temporary.replace(root/'queue-status.json')
    if not pending and not running: break
    if resources_ok():
        snapshot=gpu_snapshot()
        for gpu in a.gpus.split(','):
            if gpu in running or gpu not in snapshot: continue
            uuid, used, utilization=snapshot[gpu]
            # A zero-utilization card with an allocated process is still occupied.
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            if uuid in apps or int(used)>128 or int(utilization)>5: continue
            available=[j for j in pending if ready(j)]
            if not available: continue
            job=available[0]; pending.remove(job)
            out=root/'runs'/run_id(job); out.mkdir(parents=True,exist_ok=True)
            log=(out/'console.log').open('a')
            cmd=['nice','-n','10',sys.executable,'-u',str(repo/'src/run_task4.py'),
                 '--dataset',job['dataset'],'--variant',job['variant'],'--seed',str(job['seed']),
                 '--gpu',gpu,'--data-root',str(root/'data'),'--output',str(out)]
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=repo/'src')
            running[gpu]=(proc,log,job)
            print('START',run_id(job),'GPU',gpu,'PID',proc.pid,flush=True)
    if failed:
        print('Failed jobs require review:',sorted(failed),flush=True)
        # Do not spread an implementation failure over the remaining matrix.
        for proc,log,job in running.values():
            proc.wait(); log.close()
        break
    time.sleep(30)
subprocess.run([sys.executable,str(repo/'task4/summarize.py'),'--root',str(root)])
