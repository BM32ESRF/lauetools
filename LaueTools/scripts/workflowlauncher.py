r"""
workflowlauncher: run LaueTools workflows (ROI counters and mosaic, see workflows.ROICountersWorkflow)
described by a configuration file, on the local machine or on a SLURM partition of ESRF cluster.

The same workflow can be launched from the LaueTools GUI (PeakSearchGUI 'Mosaic & Monitor' tab),
from a jupyter notebook or from the command line::

    # compute in this process (e.g. inside a SLURM job or on a powerful machine)
    python -m LaueTools.scripts.workflowlauncher run config.json --ncpus 32
    # launch in background on this machine or submit to a SLURM partition
    python -m LaueTools.scripts.workflowlauncher submit config.json --resource magnifix --ncpus 96 --wait
    # list computing resources
    python -m LaueTools.scripts.workflowlauncher resources

config.json::

    {"workflow": "roi_counters",
     "output": "/path/to/results.h5",
     "params": {...  see workflows.ROICountersWorkflow  ...}}

From a notebook::

    import LaueTools.scripts.workflowlauncher as WL
    config = WL.make_config(params, output='/path/to/results.h5')
    job = WL.launch(config, resource='magnifix', nbcpus=96)   # or resource='local'
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

import LaueTools
import LaueTools.scripts.workflows as wf

# 'type': 'local' (this machine) or 'slurm' (partition, optional constraint)
# 'nbcpus': default nb of cpus
COMPUTING_RESOURCES = {
    'local': {'type': 'local', 'description': 'this machine'},
    'magnifix': {'type': 'slurm', 'partition': 'magnifix', 'nbcpus': 96,
                    'description': 'SLURM partition magnifix'},
    'ub20': {'type': 'slurm', 'partition': 'ub20', 'nbcpus': 96,
                'description': 'SLURM partition ub20'},
    'nice-hpc7': {'type': 'slurm', 'partition': 'nice', 'constraint': 'hpc7', 'nbcpus': 96,
                    'description': 'SLURM partition nice, hpc7 nodes'},
    'nice-hpc8': {'type': 'slurm', 'partition': 'nice', 'constraint': 'hpc8', 'nbcpus': 64,
                    'description': 'SLURM partition nice, hpc8 nodes'},
    'nice-hpc6': {'type': 'slurm', 'partition': 'nice', 'constraint': 'hpc6', 'nbcpus': 40,
                    'description': 'SLURM partition nice, hpc6 nodes'},
}
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


def run_config(config, nb_cpus: Optional[int] = None) -> Dict[str, Any]:
    """compute workflow described by config (dict or path to json file) in this process,
    save results in config['output'] and update job status file"""
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    workflow = config.get('workflow', 'roi_counters')
    output = config['output']
    statuspath = job_filepaths(output)['status']
    jobinfo = {'host': socket.gethostname(), 'pid': os.getpid(),
                'slurm_jobid': os.environ.get('SLURM_JOB_ID')}
    write_status(statuspath, state='RUNNING', done=0, total=None, **jobinfo)
    try:
        if workflow != 'roi_counters':
            raise ValueError(f"unknown workflow '{workflow}'")
        params = config['params']

        def progress(done, total):
            write_status(statuspath, state='RUNNING', done=done, total=total, **jobinfo)

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


def _slurm_command(args, submit_host: Optional[str] = None):
    """SLURM command args, run through ssh on submit host if not available on this machine"""
    if shutil.which(args[0]):
        return list(args)
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', submit_host or SLURM_SUBMIT_HOST,
            ' '.join(shlex.quote(str(arg)) for arg in args)]


def slurm_script(configpath: str, resource: Dict[str, Any], nbcpus: int, jobname: str,
                    time_limit: str = DEFAULT_TIME_LIMIT, mem_per_cpu: str = DEFAULT_MEM_PER_CPU,
                    python: Optional[str] = None) -> str:
    """content of sbatch script running workflow of configpath"""
    configpath = os.path.abspath(configpath)
    with open(configpath, 'r') as f:
        files = job_filepaths(json.load(f)['output'])
    python = python or sys.executable
    lines = ['#!/bin/bash -l',
             f'#SBATCH --job-name={jobname}',
             f"#SBATCH --partition={resource['partition']}"]
    if resource.get('constraint'):
        lines.append(f"#SBATCH --constraint={resource['constraint']}")
    lines += ['#SBATCH --nodes=1',
              '#SBATCH --ntasks=1',
              f'#SBATCH --cpus-per-task={nbcpus}',
              f'#SBATCH --mem-per-cpu={mem_per_cpu}',
              f'#SBATCH --time={time_limit}',
              f"#SBATCH --output={files['log']}",
              '',
              'echo "host: $(hostname)   date: $(date)   cpus: $SLURM_CPUS_PER_TASK"',
              f'module load mamba > /dev/null 2>&1 && conda activate {shlex.quote(sys.prefix)} > /dev/null 2>&1',
              'export OMP_NUM_THREADS=1',
              f'export PYTHONPATH={shlex.quote(LAUETOOLS_PARENT_FOLDER)}${{PYTHONPATH:+:$PYTHONPATH}}',
              f'{shlex.quote(python)} -m LaueTools.scripts.workflowlauncher run '
              f"{shlex.quote(configpath)} --ncpus ${{SLURM_CPUS_PER_TASK:-{nbcpus}}}",
              'rc=$?',
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
        if self.process is not None and status['state'] not in FINAL_STATES:
            returncode = self.process.poll()
            if returncode is not None:
                # process ended without writing final state (e.g. killed)
                status = read_status(self.files['status']) or status
                if status['state'] not in FINAL_STATES:
                    status = {'state': 'FAILED', 'message': f'process exit code {returncode}, see log'}
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
        """load results (see workflows.load_results())"""
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
            submit_host: Optional[str] = None, python: Optional[str] = None) -> WorkflowJob:
    """launch workflow of config (see make_config()) in background on this machine
    (resource='local') or submit it to a SLURM partition (see COMPUTING_RESOURCES)

    :param nbcpus: nb of cpus (default: nbcpus of resource or all cpus of local machine)
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
    nbcpus = int(nbcpus or resource_def.get('nbcpus') or wf.available_cpus())
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
        f.write(slurm_script(configpath, resource_def, nbcpus, jobname, time_limit, mem_per_cpu, python))
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
    write_status(files['status'], state='PENDING', resource=resource, slurm_jobid=jobid)
    print(f'SLURM job {jobid} submitted on {resource}, log: {files["log"]}')
    return WorkflowJob(output, resource, jobid=jobid, submit_host=submit_host)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m LaueTools.scripts.workflowlauncher',
                                     description='run LaueTools workflow (ROI counters, mosaic) '
                                     'described in a json configuration file')
    subparsers = parser.add_subparsers(dest='command', required=True)
    prun = subparsers.add_parser('run', help='compute workflow in this process')
    prun.add_argument('config', help='json configuration file')
    prun.add_argument('--ncpus', type=int, default=None, help='nb of cpus (default: all available)')

    psubmit = subparsers.add_parser('submit', help='launch workflow in background on this machine '
                                                    'or submit it to a SLURM partition')
    psubmit.add_argument('config', help='json configuration file')
    psubmit.add_argument('--resource', default='local', choices=list(COMPUTING_RESOURCES),
                            help='computing resource (default: local)')
    psubmit.add_argument('--ncpus', type=int, default=None)
    psubmit.add_argument('--time', default=DEFAULT_TIME_LIMIT, help='SLURM time limit')
    psubmit.add_argument('--mem-per-cpu', default=DEFAULT_MEM_PER_CPU, help='SLURM memory per cpu')
    psubmit.add_argument('--submit-host', default=None,
                            help=f'host to submit SLURM job through ssh (default: {SLURM_SUBMIT_HOST})')
    psubmit.add_argument('--wait', action='store_true', help='wait for the end of the job')

    subparsers.add_parser('resources', help='list computing resources')

    args = parser.parse_args(argv)
    if args.command == 'run':
        run_config(args.config, args.ncpus)
    elif args.command == 'submit':
        with open(args.config, 'r') as f:
            config = json.load(f)
        job = launch(config, args.resource, args.ncpus, args.time, args.mem_per_cpu, args.submit_host)
        if args.wait:
            status = job.wait()
            if status['state'] != 'COMPLETED':
                print(job.log_tail())
                sys.exit(1)
    elif args.command == 'resources':
        for name, resource in COMPUTING_RESOURCES.items():
            print(f"{name:12s} {resource.get('description', '')}"
                    + (f" ({resource['nbcpus']} cpus)" if resource.get('nbcpus') else ''))


if __name__ == '__main__':
    main()
