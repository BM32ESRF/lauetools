r"""
workflowlauncher: run LaueTools workflows (ROI counters and mosaic, see workflows.ROICountersWorkflow;
quick daxm, see daxm_workflow) described by a configuration file, on the local machine or on a SLURM partition of ESRF cluster.

The same workflow can be launched from the LaueTools GUI (PeakSearchGUI 'Mosaic & Monitor' tab),
from a jupyter notebook or from the command line::

    # compute in this process (e.g. inside a SLURM job or on a powerful machine)
    python -m LaueTools.scripts.workflowlauncher run config.json --ncpus 32
    # launch in background on this machine or submit to a SLURM partition
    python -m LaueTools.scripts.workflowlauncher submit config.json --resource magnifix --wait   # whole node
    # on several nodes (images shared between nodes, results merged at the end of the job)
    python -m LaueTools.scripts.workflowlauncher submit config.json --resource magnifix3 --wait
    # list computing resources
    python -m LaueTools.scripts.workflowlauncher resources

config.json::

    {"workflow": "roi_counters",
     "output": "/path/to/results.h5",
     "params": {...  see workflows.ROICountersWorkflow  ...}}

    "roicenters" in params: list of [X, Y], peak list file path (.dat, .cor, .fit, absolute path for
    SLURM jobs) or grid {"grid": step} (see workflows.resolve_roicenters())

From a notebook::

    import LaueTools.scripts.workflowlauncher as WL
    config = WL.make_config(params, output='/path/to/results.h5')
    job = WL.launch(config, resource='magnifix')   # whole node of 192 cpus; or resource='local', nbcpus=32
    job.wait()
    results = job.results()

Job files are written next to the output file (results.config.json, results.log,
results.status.json and results.slurm). Job state is read in results.status.json written by the
running workflow (on /data shared by all ESRF machines), so no SLURM command is needed to follow it.

Computing resources are defined in COMPUTING_RESOURCES and can be completed or modified by the
user in ~/.lauetools/computing_resources.json (same structure).
SLURM commands (sbatch, scancel) are run through ssh on SLURM_SUBMIT_HOST when they are not
available on this machine (environment variable LAUETOOLS_SLURM_SUBMIT_HOST).
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

import LaueTools
import LaueTools.scripts.workflows as wf

# 'type': 'local' (this machine) or 'slurm' (partition, optional constraint)
# 'nbcpus': default nb of cpus (per node). Not set for 'nodes' > 1: whole nodes (all their cpus)
# 'nodecpus': nb of cpus of a node, for whole nodes (default: from sinfo when the job is submitted)
# 'nodes': nb of nodes (default 1). Images are shared between nodes (1 task per node, part files of
#     results merged at the end of the job, see run_config() and merge_parts())
# 'whole_nodes': jobs always ask for all the cpus of their nodes, whatever nbcpus is given (ESRF IT: any node of
#     magnifix is given only to jobs of 192 cpus per node: 192 for magnifix, 2 x 192 for magnifix2, 3 x 192 for
#     magnifix3). magnifix2 and magnifix3 are the partition magnifix with 2 or 3 nodes (sinfo -s: no other partition)
COMPUTING_RESOURCES = {
    'local': {'type': 'local', 'description': 'this machine'},
    'magnifix': {'type': 'slurm', 'partition': 'magnifix', 'nodecpus': 192, 'whole_nodes': True,
                    'description': 'SLURM partition magnifix, 1 whole node (192 cpus)'},
    'magnifix2': {'type': 'slurm', 'partition': 'magnifix', 'nodes': 2, 'nodecpus': 192, 'whole_nodes': True,
                    'description': 'SLURM partition magnifix, 2 whole nodes (2 x 192 cpus, images shared between nodes)'},
    'magnifix3': {'type': 'slurm', 'partition': 'magnifix', 'nodes': 3, 'nodecpus': 192, 'whole_nodes': True,
                    'description': 'SLURM partition magnifix, 3 whole nodes (3 x 192 cpus, images shared between nodes)'},
    'ub20': {'type': 'slurm', 'partition': 'ub20', 'nbcpus': 96,
                'description': 'SLURM partition ub20'},
    'nice-hpc7': {'type': 'slurm', 'partition': 'nice', 'constraint': 'hpc7', 'nbcpus': 96,
                    'description': 'SLURM partition nice, hpc7 nodes'},
    'nice-hpc8': {'type': 'slurm', 'partition': 'nice', 'constraint': 'hpc8', 'nbcpus': 64,
                    'description': 'SLURM partition nice, hpc8 nodes'},
    'nice-hpc6': {'type': 'slurm', 'partition': 'nice', 'constraint': 'hpc6', 'nbcpus': 40,
                    'description': 'SLURM partition nice, hpc6 nodes'},
}
# cpus of this machine available for LaueTools (cpu quota of a VISA instance or container, SLURM allocation)
COMPUTING_RESOURCES['local']['description'] = f'this machine ({wf.available_cpus()} cpus)'
USER_RESOURCES_FILE = os.path.join(os.path.expanduser('~'), '.lauetools', 'computing_resources.json')
try:
    with open(USER_RESOURCES_FILE, 'r') as _f:
        COMPUTING_RESOURCES.update(json.load(_f))
except (OSError, ValueError):
    pass

SLURM_SUBMIT_HOST = os.environ.get('LAUETOOLS_SLURM_SUBMIT_HOST', 'cluster-access')
DEFAULT_TIME_LIMIT = '01:00:00'
DEFAULT_MEM_PER_CPU = '1G'

FINAL_STATES = ('COMPLETED', 'FAILED', 'CANCELLED')

# folder containing LaueTools package (for PYTHONPATH of jobs)
LAUETOOLS_PARENT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(LaueTools.__file__)))


def job_filepaths(output: str) -> Dict[str, str]:
    """paths of files of the job writing results in output (.h5 file)"""
    base = os.path.splitext(output)[0]
    return {'config': base + '.config.json', 'log': base + '.log',
            'status': base + '.status.json', 'script': base + '.slurm'}


def part_output(output: str, part: int, nbparts: int) -> str:
    """results file of part (task on a node) of a job computed by nbparts tasks"""
    return f'{os.path.splitext(output)[0]}.part{part}of{nbparts}.h5'


def part_config(config: Dict[str, Any], part: int, nbparts: int) -> Dict[str, Any]:
    """configuration of part of workflow: part-th chunk of images (contiguous, map order)"""
    if not 0 <= part < nbparts:
        raise ValueError(f'part {part} not in [0, {nbparts - 1}]')
    listindices = np.array_split(np.ravel(config['params']['listindices']).astype(int), nbparts)[part]
    params = dict(config['params'], listindices=listindices.tolist())
    return dict(config, params=params, output=part_output(config['output'], part, nbparts))


def write_status(path: str, **status):
    """write job status (json) in file path (state: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)"""
    status['date'] = time.strftime('%Y-%m-%d %H:%M:%S')
    tmppath = f'{path}.{os.getpid()}.tmp'
    with open(tmppath, 'w') as f:
        json.dump(status, f)
    os.replace(tmppath, path)


def read_status(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# where to write results of images of a folder: 'image' (folder of images, default),
# 'mirror' (mirror folder in PROCESSED_DATA of images folder in RAW_DATA) or path of a folder
OUTPUT_FOLDER_MODES = ('image', 'mirror')
FALLBACK_OUTPUT_FOLDER = os.path.join(os.path.expanduser('~'), 'lauetools_results')


def mirror_folder(imagefolder: str) -> Optional[str]:
    """mirror folder in PROCESSED_DATA of imagefolder in RAW_DATA
    (.../RAW_DATA/{sample}/{sample}_{dataset}/scanXXXX -> .../PROCESSED_DATA/{sample}/{sample}_{dataset}/scanXXXX),
    None if imagefolder is not in RAW_DATA"""
    parts = os.path.abspath(imagefolder).split(os.sep)
    if 'RAW_DATA' not in parts:
        return None
    irawdata = len(parts) - 1 - parts[::-1].index('RAW_DATA')
    return os.sep.join(parts[:irawdata] + ['PROCESSED_DATA'] + parts[irawdata + 1:])


def _writable(folder: str) -> bool:
    """folder is writable or can be created (first existing parent folder is writable)"""
    while folder and not os.path.isdir(folder):
        parent = os.path.dirname(folder)
        if parent == folder:
            return False
        folder = parent
    return os.access(folder, os.W_OK)


def resolve_output_folder(imagefolder: str, mode: str = 'image') -> Tuple[str, str]:
    """folder where to write results of images of imagefolder, without creating it

    :param mode: 'image' (imagefolder), 'mirror' (PROCESSED_DATA mirror folder, see mirror_folder())
        or path of a folder
    :return: (folder, note) note explains a fallback ('' if requested folder is used): to the
        PROCESSED_DATA mirror folder if folder is not writable, then to FALLBACK_OUTPUT_FOLDER
    """
    mirror = mirror_folder(imagefolder)
    if mode == 'image':
        candidates = [(imagefolder, ''), (mirror, 'image folder not writable: PROCESSED_DATA mirror used')]
    elif mode == 'mirror':
        candidates = [(mirror, ''), (imagefolder, 'images not in RAW_DATA: image folder used')]
    else:
        candidates = [(mode, ''), (mirror, f'{mode} not writable: PROCESSED_DATA mirror used')]
    for folder, note in candidates:
        if folder and _writable(folder):
            return folder, note
    return FALLBACK_OUTPUT_FOLDER, 'requested folder not writable: home folder used'


# folders visible from SLURM cluster nodes (job files and results of SLURM jobs)
SHARED_FOLDERS = ('/data/', '/gpfs/', '/mnt/multipath-shares/')


def is_shared_folder(folder: str) -> bool:
    """folder is visible from SLURM cluster nodes"""
    return os.path.abspath(folder).rstrip(os.sep).startswith(tuple(p.rstrip('/') for p in SHARED_FOLDERS))


def output_folder(imagefolder: str, mode: str = 'image') -> str:
    """create (if needed) and return folder where to write results of images of imagefolder
    (see resolve_output_folder())"""
    folder, note = resolve_output_folder(imagefolder, mode)
    if note:
        print(f'output folder: {note}')
    os.makedirs(folder, exist_ok=True)
    return folder


def make_config(params: Dict[str, Any], output: str, workflow: str = 'roi_counters') -> Dict[str, Any]:
    """configuration of workflow (params: see workflows.ROICountersWorkflow, output: .h5 file,
    None for InSessionJob without results file)"""
    return {'workflow': workflow, 'output': None if output is None else os.path.abspath(output),
            'params': wf.to_jsonable(params)}


def write_config(config: Dict[str, Any], path: Optional[str] = None) -> str:
    if path is None:
        path = job_filepaths(config['output'])['config']
    with open(path, 'w') as f:
        json.dump(wf.to_jsonable(config), f, indent=1)
    return path


# workflows of run_config(): 'roi_counters' (workflows.ROICountersWorkflow),
# 'quickdaxm' (daxm_workflow.compute_quickdaxm, params: daxm configuration)
WORKFLOWS = ('roi_counters', 'quickdaxm')


def _run_quickdaxm(params, output, nb_cpus, progress, statuspath, jobinfo) -> Dict[str, Any]:
    # imported here: DAXM modules (numba) are not needed by the other workflows
    import LaueTools.daxm_workflow as DW
    from LaueTools.Daxm import quickdaxm as QD
    results = DW.compute_quickdaxm(params, nb_cpus, progress)
    QD.save_results(results, output)
    nbimages = len(results['missing'])
    write_status(statuspath, state='COMPLETED', done=nbimages, total=nbimages, output=output,
                    nbmissing=int(results['missing'].sum()), **jobinfo)
    print(f'results saved in {output}', flush=True)
    return results


def read_config(config) -> Dict[str, Any]:
    if isinstance(config, str):
        with open(config, 'r') as f:
            return json.load(f)
    return config


def run_config(config, nb_cpus: Optional[int] = None, part: Optional[int] = None,
                nbparts: int = 1) -> Dict[str, Any]:
    """compute workflow described by config (dict or path to json file) in this process,
    save results in config['output'] and update job status file

    :param part, nbparts: compute only part-th chunk of images out of nbparts (results in
        part_output() file, see merge_parts()). Default part: SLURM task index (SLURM_PROCID)
    """
    config = read_config(config)
    if nbparts > 1:
        if part is None:
            part = int(os.environ.get('SLURM_PROCID', 0))
        config = part_config(config, part, nbparts)
    workflow = config.get('workflow', 'roi_counters')
    output = config['output']
    statuspath = job_filepaths(output)['status']
    jobinfo = {'host': socket.gethostname(), 'pid': os.getpid(),
                'slurm_jobid': os.environ.get('SLURM_JOB_ID')}
    write_status(statuspath, state='RUNNING', done=0, total=None, **jobinfo)
    try:
        if workflow not in WORKFLOWS:
            raise ValueError(f"unknown workflow '{workflow}'")
        params = config['params']

        def progress(done, total):
            write_status(statuspath, state='RUNNING', done=done, total=total, **jobinfo)

        if workflow == 'quickdaxm':
            return _run_quickdaxm(params, output, nb_cpus, progress, statuspath, jobinfo)
        # progress bar in a terminal, progress lines in log file of job
        results = wf.ROICountersWorkflow(params).run_roi_counters_workflow(nb_cpus,
                                                    progressbar=sys.stderr.isatty(),
                                                    progress_callback=progress)
        wf.save_results(results, output)
    except BaseException as exc:
        write_status(statuspath, state='FAILED', message=f'{type(exc).__name__}: {exc}', **jobinfo)
        raise
    nbimages = len(results['imageindices'])
    write_status(statuspath, state='COMPLETED', done=nbimages, total=nbimages, output=output,
                    nbmissing=int(results['missing'].sum()), **jobinfo)
    print(f'results saved in {output}', flush=True)
    return results


def merge_parts(config, nbparts: int, remove: bool = True) -> Dict[str, Any]:
    """merge results of the nbparts parts of workflow of config (see run_config()) in config['output']
    and update job status file

    :param remove: remove part files (results and status) after merging
    """
    config = read_config(config)
    output = config['output']
    statuspath = job_filepaths(output)['status']
    jobinfo = {'host': socket.gethostname(), 'slurm_jobid': os.environ.get('SLURM_JOB_ID')}
    try:
        if config.get('workflow', 'roi_counters') != 'roi_counters':
            raise ValueError(f"merging of parts not implemented for workflow '{config.get('workflow')}'")
        parts = [wf.load_results(part_output(output, k, nbparts)) for k in range(nbparts)]
        results = {key: np.concatenate([part[key] for part in parts])
                    for key in parts[0] if key != 'params'}
        results['params'] = wf.ROICountersWorkflow(config['params']).params
        wf.save_results(results, output)
    except BaseException as exc:
        write_status(statuspath, state='FAILED', message=f'merge: {type(exc).__name__}: {exc}', **jobinfo)
        raise
    if remove:
        for k in range(nbparts):
            for path in (part_output(output, k, nbparts),
                         job_filepaths(part_output(output, k, nbparts))['status']):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
    nbimages = len(results['imageindices'])
    write_status(statuspath, state='COMPLETED', done=nbimages, total=nbimages, output=output,
                    nbmissing=int(results['missing'].sum()), **jobinfo)
    print(f'{nbparts} parts merged, results saved in {output}', flush=True)
    return results


def _slurm_command(args, submit_host: Optional[str] = None):
    """SLURM command args, run through ssh on submit host if not available on this machine"""
    if shutil.which(args[0]):
        return list(args)
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', submit_host or SLURM_SUBMIT_HOST,
            ' '.join(shlex.quote(str(arg)) for arg in args)]


def partition_nodecpus(resource: Dict[str, Any], submit_host: Optional[str] = None) -> Optional[int]:
    """nb of cpus of the nodes of the partition (and constraint) of a SLURM resource: 'nodecpus' of a resource of
    whole nodes (magnifix: 192, its smaller GPU nodes are not used), else sinfo (smallest node), else 'nodecpus' of
    the resource, None if unknown"""
    if resource.get('whole_nodes') and resource.get('nodecpus'):
        return int(resource['nodecpus'])
    args = ['sinfo', '-h', '-p', resource['partition'], '-o', '%c %f']
    try:
        res = subprocess.run(_slurm_command(args, submit_host), capture_output=True, text=True, timeout=30)
        cpus = [int(line.split()[0].rstrip('+')) for line in res.stdout.splitlines()
                if line.split() and line.split()[0].rstrip('+').isdigit()
                and (not resource.get('constraint') or resource['constraint'] in line)]
        if res.returncode == 0 and cpus:
            return min(cpus)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return resource.get('nodecpus')


def slurm_script(configpath: str, resource: Dict[str, Any], nbcpus: Optional[int], jobname: str,
                    time_limit: str = DEFAULT_TIME_LIMIT, mem_per_cpu: str = DEFAULT_MEM_PER_CPU,
                    python: Optional[str] = None, nbnodes: int = 1, nodecpus: Optional[int] = None) -> str:
    """content of sbatch script running workflow of configpath

    :param nbcpus: nb of cpus per node (None: whole nodes, all their cpus and memory)
    :param nodecpus: nb of cpus of a node, for whole nodes (nbcpus None): --cpus-per-task and --mem=0 (the
        cluster rejects --exclusive). Default: 'nodecpus' of resource
    :param nbnodes: nb of nodes. For nbnodes > 1, images are shared between nodes (1 task per node,
        see run_config()) and the parts of results are merged at the end (see merge_parts())
    """
    configpath = os.path.abspath(configpath)
    with open(configpath, 'r') as f:
        files = job_filepaths(json.load(f)['output'])
    python = python or sys.executable
    lines = ['#!/bin/bash -l',
             f'#SBATCH --job-name={jobname}',
             f"#SBATCH --partition={resource['partition']}"]
    if resource.get('constraint'):
        lines.append(f"#SBATCH --constraint={resource['constraint']}")
    lines += [f'#SBATCH --nodes={nbnodes}',
              f'#SBATCH --ntasks={nbnodes}',
              '#SBATCH --ntasks-per-node=1']
    if nbcpus:
        lines += [f'#SBATCH --cpus-per-task={nbcpus}',
                  f'#SBATCH --mem-per-cpu={mem_per_cpu}']
        taskcpus = f'${{SLURM_CPUS_PER_TASK:-{nbcpus}}}'
    else:  # whole nodes: all their cpus and all their memory (--exclusive is rejected by the ESRF cluster)
        nodecpus = nodecpus or resource.get('nodecpus')
        if not nodecpus:
            raise ValueError(f"nb of cpus per node of {resource['partition']} unknown: give nbcpus, or 'nodecpus' "
                             'in the computing resource')
        lines += [f'#SBATCH --cpus-per-task={int(nodecpus)}',
                  '#SBATCH --mem=0']
        taskcpus = f'${{SLURM_CPUS_PER_TASK:-{int(nodecpus)}}}'
    lines += [f'#SBATCH --time={time_limit}',
              f"#SBATCH --output={files['log']}",
              '',
              f'echo "host: $(hostname)   date: $(date)   nodes: $SLURM_JOB_NODELIST   cpus per node: {taskcpus}"',
              f'module load mamba > /dev/null 2>&1 && conda activate {shlex.quote(sys.prefix)} > /dev/null 2>&1',
              'export OMP_NUM_THREADS=1',
              f'export PYTHONPATH={shlex.quote(LAUETOOLS_PARENT_FOLDER)}${{PYTHONPATH:+:$PYTHONPATH}}']
    launcher = f'{shlex.quote(python)} -m LaueTools.scripts.workflowlauncher'
    if nbnodes == 1:
        lines += [f'{launcher} run {shlex.quote(configpath)} --ncpus {taskcpus}',
                  'rc=$?']
    else:
        # 1 task per node (part index: SLURM_PROCID), then merge of parts
        lines += [f'srun --label --kill-on-bad-exit=1 --cpus-per-task={taskcpus} '
                  f'{launcher} run {shlex.quote(configpath)} --ncpus {taskcpus} --nbparts {nbnodes}',
                  'rc=$?',
                  f'if [ $rc -eq 0 ]; then {launcher} merge {shlex.quote(configpath)} --nbparts {nbnodes}; rc=$?; fi']
    lines += [
              # failure before python could write the job status (e.g. environment problem)
              f"if [ $rc -ne 0 ] && ! grep -q FAILED {shlex.quote(files['status'])} 2>/dev/null; then",
              f"""  echo '{{"state": "FAILED", "message": "exit code '$rc', see log"}}' > {shlex.quote(files['status'])}""",
              'fi',
              'exit $rc',
              '']
    return '\n'.join(lines)


class WorkflowJob:
    """workflow launched by launch() on the local machine or on a SLURM partition"""

    def __init__(self, output: str, resource: str = 'local', jobid: Optional[str] = None,
                    process: Optional[subprocess.Popen] = None, submit_host: Optional[str] = None):
        self.output = output
        self.resource = resource
        self.jobid = jobid
        self.process = process
        self.submit_host = submit_host
        self.files = job_filepaths(output)

    @property
    def is_slurm(self) -> bool:
        return COMPUTING_RESOURCES.get(self.resource, {}).get('type') == 'slurm'

    def status(self) -> Dict[str, Any]:
        """job status dict with key 'state' (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
        and, while running, 'done' and 'total' (nb of processed images and nb of images)"""
        status = read_status(self.files['status']) or {'state': 'PENDING'}
        if status['state'] == 'RUNNING' and status.get('nbparts'):
            status = self._parts_status(status)
        if self.process is not None and status['state'] not in FINAL_STATES:
            returncode = self.process.poll()
            if returncode is not None:
                # process ended without writing final state (e.g. killed)
                status = read_status(self.files['status']) or status
                if status['state'] not in FINAL_STATES:
                    status = {'state': 'FAILED', 'message': f'process exit code {returncode}, see log'}
        return status

    def _parts_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """status of a job computed by several tasks (status files of parts, see run_config())"""
        nbparts = status['nbparts']
        parts = [read_status(job_filepaths(part_output(self.output, k, nbparts))['status']) or
                    {'state': 'PENDING'} for k in range(nbparts)]
        if all(part['state'] == 'PENDING' for part in parts):  # e.g. job in SLURM queue
            return dict(status, state='PENDING')
        failed = [part for part in parts if part['state'] == 'FAILED']
        if failed:
            return dict(failed[0], state='RUNNING', message=f"a part failed: {failed[0].get('message')}")
        totals = [part.get('total') for part in parts]
        status = dict(status, done=sum(part.get('done') or 0 for part in parts),
                        total=sum(totals) if None not in totals else None,
                        host=', '.join(part['host'] for part in parts if part.get('host')))
        if all(part['state'] == 'COMPLETED' for part in parts):
            status['message'] = 'merging parts'
        return status

    def state(self) -> str:
        return self.status()['state']

    def finished(self) -> bool:
        return self.state() in FINAL_STATES

    def wait(self, poll: float = 2., timeout: Optional[float] = None, verbose: bool = True) -> Dict[str, Any]:
        """wait for the end of the job and return its final status
        (verbose: show job progress bar, tqdm widget in jupyter notebook)"""
        t0 = time.time()
        progressbar = JobProgressBar(self) if verbose else None
        try:
            while True:
                status = self.status()
                if progressbar is not None:
                    progressbar.update(status)
                if status['state'] in FINAL_STATES:
                    return status
                if timeout is not None and time.time() - t0 > timeout:
                    return status
                time.sleep(poll)
        finally:
            if progressbar is not None:
                progressbar.close()

    def describe(self, status: Optional[Dict[str, Any]] = None) -> str:
        """one line description of job state"""
        status = status or self.status()
        where = self.resource + (f' job {self.jobid}' if self.jobid else '')
        text = f"[{where}] {status['state']}"
        if status['state'] == 'RUNNING' and status.get('total'):
            text += f" {status['done']}/{status['total']} images"
            if status.get('host'):
                text += f" on {status['host']}"
        if status.get('message'):
            text += f": {status['message']}"
        return text

    def results(self) -> Dict[str, Any]:
        """load results (see workflows.load_results(), or quickdaxm.load_results() for 'quickdaxm')"""
        try:
            with open(self.files['config'], 'r') as f:
                workflow = json.load(f).get('workflow', 'roi_counters')
        except (OSError, ValueError):
            workflow = 'roi_counters'
        if workflow == 'quickdaxm':
            from LaueTools.Daxm import quickdaxm as QD
            return QD.load_results(self.output)
        return wf.load_results(self.output)

    def remove_files(self):
        """remove results file and job files (config, log, status, script)"""
        for path in [self.output] + list(self.files.values()):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def log_tail(self, nblines: int = 20) -> str:
        try:
            with open(self.files['log'], 'r') as f:
                return ''.join(f.readlines()[-nblines:])
        except OSError:
            return ''

    def cancel(self):
        """stop the job"""
        if self.process is not None and self.process.poll() is None:
            if hasattr(os, 'killpg'):  # process and its pool of workers
                os.killpg(self.process.pid, signal.SIGTERM)
            else:
                self.process.terminate()
        elif self.is_slurm and self.jobid:
            subprocess.run(_slurm_command(['scancel', self.jobid], self.submit_host),
                            capture_output=True, text=True, timeout=30)
        write_status(self.files['status'], state='CANCELLED')

    def __repr__(self):
        return f'WorkflowJob({self.describe()}, output={self.output})'


class InSessionJob(WorkflowJob):
    """workflow computed in a thread of this python session (e.g. GUI), images being processed by
    workers of workflows.session_mp_context(): no new python process importing LaueTools at each
    launch. Results are kept in memory and saved in output file only if output is not None.
    Same interface as WorkflowJob (status(), results(), cancel(), ...)."""

    def __init__(self, config: Dict[str, Any], nbcpus: Optional[int] = None):
        output = config.get('output')
        WorkflowJob.__init__(self, output or '', resource='local')
        if output is None:
            self.files = {}
        self.config = config
        self._status = {'state': 'PENDING'}
        self._results = None
        self.workflow = wf.ROICountersWorkflow(config['params'])
        self.thread = threading.Thread(target=self._run, args=(nbcpus,), daemon=True)
        self.thread.start()

    def _run(self, nbcpus):
        host = socket.gethostname()
        self._status = {'state': 'RUNNING', 'done': 0, 'total': None, 'host': host}

        def progress(done, total):
            self._status = {'state': 'RUNNING', 'done': done, 'total': total, 'host': host}

        try:
            results = self.workflow.run_roi_counters_workflow(nbcpus, progress_callback=progress,
                                                    mp_context=wf.session_mp_context())
            if self.output:
                write_config(self.config, self.files['config'])
                wf.save_results(results, self.output)
                print(f'results saved in {self.output}')
        except wf.WorkflowCancelled:
            self._status = {'state': 'CANCELLED'}
            return
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._status = {'state': 'FAILED', 'message': f'{type(exc).__name__}: {exc}'}
            return
        self._results = results
        nbimages = len(results['imageindices'])
        self._status = {'state': 'COMPLETED', 'done': nbimages, 'total': nbimages, 'host': host}

    def status(self) -> Dict[str, Any]:
        return dict(self._status)

    def results(self) -> Dict[str, Any]:
        return self._results

    def cancel(self):
        self.workflow.cancel_requested = True
        self._status = {'state': 'CANCELLED'}

    def remove_files(self):
        if self.output:
            WorkflowJob.remove_files(self)

    def log_tail(self, nblines: int = 20) -> str:
        return 'see terminal of this python session'


class JobProgressBar:
    """tqdm progress bar (in terminal or jupyter notebook) of a WorkflowJob, updated with its status
    (see WorkflowJob.wait() and mosaic.WorkflowJobMonitor)"""

    def __init__(self, job: WorkflowJob):
        self.job = job
        self.bar = None
        self.laststate = None

    def update(self, status: Dict[str, Any]):
        state = status['state']
        if self.bar is None and status.get('total'):
            # initial: images processed before first update (correct processing rate)
            self.bar = wf.tqdm(total=status['total'], initial=status.get('done') or 0, unit='image',
                                desc=self.job.resource
                                + (f' job {self.job.jobid}' if self.job.jobid else ''))
        if self.bar is not None:
            self.bar.n = status.get('done') or 0
            self.bar.set_postfix_str(state if state != 'RUNNING' else status.get('host', ''), refresh=False)
            self.bar.refresh()
        elif state != self.laststate:  # before progress is known (e.g. job pending in SLURM queue)
            print(self.job.describe(status), flush=True)
        if state in FINAL_STATES and state != 'COMPLETED' and state != self.laststate:
            self.close()
            print(self.job.describe(status), flush=True)
        self.laststate = state

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def launch(config: Dict[str, Any], resource: str = 'local', nbcpus: Optional[int] = None,
            time_limit: str = DEFAULT_TIME_LIMIT, mem_per_cpu: str = DEFAULT_MEM_PER_CPU,
            submit_host: Optional[str] = None, python: Optional[str] = None,
            nbnodes: Optional[int] = None) -> WorkflowJob:
    """launch workflow of config (see make_config()) in background on this machine
    (resource='local') or submit it to a SLURM partition (see COMPUTING_RESOURCES)

    :param nbcpus: nb of cpus, per node for SLURM (default: nbcpus of resource or all cpus of local
        machine, or all cpus of nodes for SLURM resource of several nodes without nbcpus)
    :param nbnodes: nb of SLURM nodes (default: 'nodes' of resource, 1)
    :param time_limit, mem_per_cpu: SLURM job time limit and memory per cpu
    :param submit_host: machine where to run SLURM commands through ssh if they are not available
        on this machine (default: SLURM_SUBMIT_HOST)
    :param python: python interpreter (default: this one; must be visible from cluster for SLURM)
    :return: WorkflowJob
    """
    if resource not in COMPUTING_RESOURCES:
        raise ValueError(f"unknown computing resource '{resource}'. Possible: {list(COMPUTING_RESOURCES)}")
    resource_def = COMPUTING_RESOURCES[resource]
    output = config['output']
    os.makedirs(os.path.dirname(output), exist_ok=True)
    files = job_filepaths(output)
    configpath = write_config(config, files['config'])
    python = python or sys.executable
    nbnodes = int(nbnodes or resource_def.get('nodes', 1)) if resource_def['type'] == 'slurm' else 1
    if resource_def.get('whole_nodes'):   # all the cpus of the nodes (nbcpus None below: whole nodes)
        if nbcpus and nbcpus != resource_def.get('nodecpus'):
            print(f"{resource}: jobs of whole nodes only ({resource_def.get('nodecpus', 'all')} cpus per node): "
                  f"nbcpus {nbcpus} not used")
        nbcpus = None
    else:
        nbcpus = nbcpus or resource_def.get('nbcpus')
        if nbcpus or nbnodes == 1:
            nbcpus = int(nbcpus or wf.available_cpus())
    write_status(files['status'], state='PENDING', resource=resource)

    if resource_def['type'] == 'local':
        env = dict(os.environ, OMP_NUM_THREADS='1')
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [LAUETOOLS_PARENT_FOLDER, env.get('PYTHONPATH')]))
        cmd = [python, '-m', 'LaueTools.scripts.workflowlauncher', 'run', configpath, '--ncpus', str(nbcpus)]
        with open(files['log'], 'w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                        cwd=os.path.dirname(output), start_new_session=hasattr(os, 'killpg'))
        print(f'workflow launched on this machine (pid {process.pid}), log: {files["log"]}')
        return WorkflowJob(output, resource, jobid=None, process=process)

    if resource_def['type'] != 'slurm':
        raise ValueError(f"unknown type of computing resource {resource_def['type']}")
    jobname = 'lt_' + os.path.splitext(os.path.basename(output))[0][:40]
    with open(files['script'], 'w') as f:
        nodecpus = None if nbcpus else partition_nodecpus(resource_def, submit_host)   # whole nodes
        f.write(slurm_script(configpath, resource_def, nbcpus, jobname, time_limit, mem_per_cpu, python,
                                nbnodes, nodecpus))
    cmd = _slurm_command(['sbatch', '--parsable', files['script']], submit_host)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        write_status(files['status'], state='FAILED', message=f'sbatch failed: {exc}')
        raise RuntimeError(f"cannot submit SLURM job with: {' '.join(cmd)}\n{exc}") from exc
    if res.returncode != 0:
        message = res.stderr.strip() or res.stdout.strip()
        write_status(files['status'], state='FAILED', message=f'sbatch failed: {message}')
        raise RuntimeError(f"cannot submit SLURM job with: {' '.join(cmd)}\n{message}")
    jobid = res.stdout.strip().split(';')[0].split()[-1]
    if nbnodes > 1:  # job status: RUNNING with progress of parts, until merged results (COMPLETED)
        write_status(files['status'], state='RUNNING', resource=resource, slurm_jobid=jobid,
                        nbparts=nbnodes)
    else:
        write_status(files['status'], state='PENDING', resource=resource, slurm_jobid=jobid)
    print(f'SLURM job {jobid} submitted on {resource}' + (f' ({nbnodes} nodes)' if nbnodes > 1 else '')
            + f', log: {files["log"]}')
    return WorkflowJob(output, resource, jobid=jobid, submit_host=submit_host)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m LaueTools.scripts.workflowlauncher',
                                     description='run LaueTools workflow (ROI counters, mosaic) '
                                     'described in a json configuration file')
    subparsers = parser.add_subparsers(dest='command', required=True)
    prun = subparsers.add_parser('run', help='compute workflow in this process')
    prun.add_argument('config', help='json configuration file')
    prun.add_argument('--ncpus', type=int, default=None, help='nb of cpus (default: all available)')
    prun.add_argument('--nbparts', type=int, default=1,
                        help='compute only a part of images out of nbparts (e.g. 1 SLURM task per node)')
    prun.add_argument('--part', type=int, default=None,
                        help='index of part (default: SLURM task index SLURM_PROCID)')
    pmerge = subparsers.add_parser('merge', help='merge results of parts (run --nbparts)')
    pmerge.add_argument('config', help='json configuration file')
    pmerge.add_argument('--nbparts', type=int, required=True)
    pmerge.add_argument('--keep-parts', action='store_true', help='do not remove part files')

    psubmit = subparsers.add_parser('submit', help='launch workflow in background on this machine '
                                                    'or submit it to a SLURM partition')
    psubmit.add_argument('config', help='json configuration file')
    psubmit.add_argument('--resource', default='local', choices=list(COMPUTING_RESOURCES),
                            help='computing resource (default: local)')
    psubmit.add_argument('--ncpus', type=int, default=None, help='nb of cpus (per node for SLURM)')
    psubmit.add_argument('--nodes', type=int, default=None,
                            help='nb of SLURM nodes (default: nodes of resource, 1)')
    psubmit.add_argument('--time', default=DEFAULT_TIME_LIMIT, help='SLURM time limit')
    psubmit.add_argument('--mem-per-cpu', default=DEFAULT_MEM_PER_CPU, help='SLURM memory per cpu')
    psubmit.add_argument('--submit-host', default=None,
                            help=f'host to submit SLURM job through ssh (default: {SLURM_SUBMIT_HOST})')
    psubmit.add_argument('--wait', action='store_true', help='wait for the end of the job')

    subparsers.add_parser('resources', help='list computing resources')

    args = parser.parse_args(argv)
    if args.command == 'run':
        run_config(args.config, args.ncpus, args.part, args.nbparts)
    elif args.command == 'merge':
        merge_parts(args.config, args.nbparts, remove=not args.keep_parts)
    elif args.command == 'submit':
        with open(args.config, 'r') as f:
            config = json.load(f)
        job = launch(config, args.resource, args.ncpus, args.time, args.mem_per_cpu, args.submit_host,
                        nbnodes=args.nodes)
        if args.wait:
            status = job.wait()
            if status['state'] != 'COMPLETED':
                print(job.log_tail())
                sys.exit(1)
    elif args.command == 'resources':
        for name, resource in COMPUTING_RESOURCES.items():
            print(f"{name:12s} {resource.get('description', '')}"
                    + (f" ({resource['nbcpus']} cpus)" if resource.get('nbcpus') else '')
                    + (f" ({resource['nodes']} nodes)" if resource.get('nodes', 1) > 1 else ''))


if __name__ == '__main__':
    main()
