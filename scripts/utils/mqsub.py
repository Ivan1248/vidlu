import sys
import os
import subprocess
import signal
import time
import re
from enum import Enum
import sshkeyboard
import threading

# https://github.com/Ivan1248/scripts/tree/master/pbs

# Usage:
# [QUEUE=<name>] [RESOURCES=select=<number>:ncpus=<number>:ngpus=<number>] python mqsub.py <command> [arguments...]
# Enqueues a job with qsub and displays its output for monitoring.
#
# [QUEUE=<name>] [RESOURCES=select=<number>:ncpus=<number>:ngpus=<number>] python mqsub.py < <commands_file_name>
# Enqueues a job array with qsub and displays its output for monitoring. The commands for individual jobs should be defined in the file <commands_file_name>.
#
# If the environment variable RESOURCES is not provided, the script defaults to RESOURCES='select=1:ncpus=16:ngpus=1'.
# If the environment variable QUEUE is not provided, the script defaults to QUEUE=gpu.
# The job runs until completed or stopped by sending SIGINT or SIGTERM.


# Usage examples:
# * single job:
# python mqsub.py run-singlegpu.sh ~/path/to/script.py --arg1 value1 --arg2 value2
#   - Enqueues a job with the default configuration (select=1:ncpus=16:ngpus=1).
# RESOURCES=select=1:ncpus=1:ngpus=0 QUEUE=cpu-test python mqsub.py eval 'for i in {1..20}; do echo -ne "$i\n"; sleep 1; done; echo'
#   - Enqueues a job that increments a number and prints it every second on a single CPU in a queue called "cpu-test".
# RESOURCES=select=1:ncpus=64:ngpus=4 python mqsub.py torchrun-singlenode.sh ~/path/to/script.py --arg1 value1 --arg2 value2
#   - Enqueues a job with 4 GPUs and 64 CPUs.
# RESOURCES=select=2:ncpus=64:ngpus=4 python mqsub.py torchrun-singlenode.sh ~/path/to/script.py --arg1 value1 --arg2 value2
#   - Enqueues a job with 4 GPUs and 64 CPUs per node on 2 nodes.
# * job array:
# python mqsub.py
#   - Enqueues a job with the default configuration (select=1:ncpus=16:ngpus=1).
#
# Note:
# - Consider increasing the number of CPUs if necessary. ncpus=16 is sometimes suboptimal. See the
#   number of CPUs and GPUs per node with `pbsnodes -aSj`.

# Command, processes, filesystem, other ###########################################################

def shell_quote(arg):
    """Adds quotes around a shell argument and escapes quotes if necessary"""

    def has_unescaped_char(string, char):
        pattern = rf"(\\*{re.escape(char)})"
        for match in re.finditer(pattern, string):
            if len(match.group()) % 2 == len(char):
                return True
        return False

    if has_unescaped_char(arg, '\''):
        if has_unescaped_char(arg, '"'):
            return '\'' + arg.replace('\'', '\\\'') + '\''
        return '"' + arg + '"'
    return '\'' + arg + '\''


def reconstruct_shell_command(args):
    return ' '.join([shell_quote(arg) for arg in args]).strip()


def run_subprocess(args):
    subprocess.run(args, check=True, universal_newlines=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)


def await_file(file_path, check_interval=1):
    def get_size(file_path):
        return os.stat(file_path).st_size if os.path.isfile(file_path) else None

    last_size, size = None, get_size(file_path)
    while size is None or size != last_size:
        if check_interval > 0:
            time.sleep(check_interval)
        last_size = size
        if check_interval > 0:
            size = get_size(file_path)


def hline(n=None, text=None):
    if n is None:
        n = os.get_terminal_size()[0]
    return '-' * n if text is None else text + ' ' + '-' * (n - len(text) - 1)


# PBS file construction ###########################################################################

def construct_pbs_file(commands, modules):
    res = f"""#!/bin/bash

#PBS -q {os.environ.get("QUEUE", "gpu")}
#PBS -l {os.environ.get("RESOURCES", "select=1:ncpus=16:ngpus=1")}"""

    if len(commands) > 1:
        res += f"\n#PBS -J 0-{len(commands)}:1"

    res += "\n"
    for m in modules:
        res += f"\nmodule load {m}"
    res += '\n'

    res += '\ncd ${PBS_O_WORKDIR:-""}'
    if len(commands) > 1:
        res += '\ncase "$PBS_ARRAY_INDEX" in'
        for i, cmd in enumerate(commands):
            res += f'\n    "{i}")'
            res += f'\n        echo Running command {i}: {cmd}'
            res += f'\n        {cmd}'
            res += f'\n        ;;'
        res += '\nesac'
    else:
        res += f'\necho Running command: {commands[0]}'
        res += '\n'
        res += f"\n{commands[0]}"
    res += '\n'
    return res


# Job monitoring and control ######################################################################



def print_monitoring_info(job_id, index=None, pbs_file_path=None, outputs=None, scroll_pos_fine=0,
                          scroll_pos_pg=0):
    finished, num_errors = False, 0

    full_job_id = job_id if index is None else job_id + f"[{index}]"

    argses = [
        (['qstat', '-sw', full_job_id], 6),
        (['qcat', full_job_id], 22),
        (['qcat', '-e', full_job_id], 4)]

    if outputs is None:
        outputs = []
    for args, tail in argses:
        scroll_total = scroll_pos_fine + scroll_pos_pg * tail // 2
        try:
            full_output = subprocess.check_output(args, universal_newlines=True,
                                                  stderr=subprocess.STDOUT)
            full_output_lines = full_output.split('\n')
            n = len(full_output_lines)
            output = '\n'.join(
                full_output_lines[-tail - scroll_total:n - scroll_total]) + '\n'
            outputs.append(
                hline(text=' '.join(args) + f' [{scroll_total}/{n}]' * (scroll_total != 0)))
            outputs.append(output)
        except subprocess.CalledProcessError as e:
            num_errors += 1
            output = str(e.output)

            outputs.append(hline(text=' '.join(args)))
            outputs.append(output)

            if args[0] == 'qcat':
                if 'Job has finished' in output:
                    finished = True
                stdout_file = get_output_file_name(job_id, pbs_file_path, index=index,
                                                   stderr=args[1] == '-e')
                alt_args = ['tail', '-n', str(tail), stdout_file]
                outputs.append(hline(text=' '.join(alt_args)))
                if os.path.isfile(stdout_file):
                    try:
                        outputs.append(subprocess.check_output(
                            alt_args, universal_newlines=True, stderr=subprocess.STDOUT))
                    except e:
                        pass
    outputs.append(
        hline(text=
              "Press Ctrl+C to stop." + (
                  "" if index is None else
                  f" Press any number from 0..{len(commands) - 1} or arrow key to change the jobe being monitored.")))
    os.system('clear')
    for output in outputs:
        print(output)

    return finished, num_errors


def stop_job(job_id, pbs_file_path):
    """Stops the job and removes the PBS file. Handles the first SIGINT/SIGTERM."""
    subprocess.run(['qdel', job_id])
    try:
        os.remove(pbs_file_path)
    except FileNotFoundError as e:
        pass


def get_output_file_name(job_id, pbs_file_path, index=None, stderr=False):
    if index is not None:
        job_id = job_id + f".{index}"
    return pbs_file_path + ('.e' if stderr else '.o') + job_id


def print_output_file(job_id, pbs_file_path, index=None):
    """Prints the whole output once the output file has been created"""
    stdout_file = get_output_file_name(job_id, pbs_file_path, index=index)
    print(f"Waiting for the output file ({stdout_file})...\nPress Ctrl+C to stop.")
    await_file(stdout_file)
    print(hline(text=f"cat {stdout_file}"))
    with open(stdout_file) as f:
        print(f.read())

    stderr_file = get_output_file_name(job_id, pbs_file_path, index=index, stderr=True)
    print(hline())
    print(f"Run 'cat {stderr_file}' for printing the error output.")
    sys.exit()


# Main

class State(Enum):
    STARTING = "starting"
    SUBMITTED = "submitted"
    FINISHED = "finished"


state = State.STARTING

# cray-pals for multinode: mpiexec --cpu-bind none
modules = [s.strip() for s in
           os.environ.get("MODULES", "scientific/pytorch/1.14.0-ngc, cray-pals").split(",")]

cmd = reconstruct_shell_command(sys.argv[1:])
commands = [] if len(cmd) == 0 else [cmd]
if not sys.stdin.isatty():
    commands += [line.strip() for line in sys.stdin if len(line.strip()) > 0]
multiple_jobs = len(commands) > 1

try:  # Enable keyboard input in case stdin redirected to a file
    sys.stdin = open("/dev/tty", "r")
except IOError:
    print("Unable to open terminal for reading")
    sys.exit(1)

# Constructs a temporary PBS file
pbs_file_content = construct_pbs_file(commands, modules=modules)
pbs_file_name = "run.pbs"
with open(pbs_file_name, 'w') as f:
    f.write(pbs_file_content)

print(hline())
print(f"{pbs_file_name}:")
print(pbs_file_content)

# Submit the PBS file using qsub and get the job ID
print(hline())
print(f"Submitting job...")
try:
    output = subprocess.check_output(['qsub', pbs_file_name], universal_newlines=True)
    print(output)
except subprocess.CalledProcessError as e:
    print(e.output)
    exit(1)

state = State.SUBMITTED

print(hline())

if multiple_jobs:
    job_id = output.split('.')[0].strip()[:-2]
    print(f"Job array ID: {job_id}")
    for i, cmd in enumerate(commands):
        print(f"Job ID: {job_id}[{i}]")
        print(cmd)
else:
    job_id = output.split('.')[0].strip()
    print(f"Job ID: {job_id}")
    print(*commands)


def stop_this_job(*args):
    stop_job(job_id + "[]" if multiple_jobs else job_id, pbs_file_name)
    sshkeyboard.stop_listening()
    global state
    state = State.FINISHED


def handle_signal(*args):
    global state
    if state == state.SUBMITTED:
        stop_this_job()
        print("Stopped.\n")
    elif state == State.FINISHED:
        sys.exit("Exited!")


for sig in [signal.SIGINT, signal.SIGTERM]:
    signal.signal(sig, handle_signal)

scroll_pos_fine = 0
scroll_pos_pg = 0
monitoring_index = 0 if multiple_jobs else None
keyboard_input_received = False


def switch_monitoring_index(key):
    try:
        global monitoring_index, scroll_pos_fine, scroll_pos_pg, keyboard_input_received

        # print(f"'{key}' pressed")

        keyboard_input_received = True
        if key in ['up', 'down']:
            scroll_pos_fine = max(scroll_pos_fine + (1 if key == 'up' else -1), 0)
        elif key == 'pagedown' and scroll_pos_pg == 0:
            scroll_pos_fine = max(scroll_pos_fine + (1 if key == 'pageup' else -1), 0)
        elif key in ['pageup', 'pagedown']:
            scroll_pos_pg = max(scroll_pos_pg + (1 if key == 'pageup' else -1), 0)
        else:
            keyboard_input_received = False

        if multiple_jobs:
            keyboard_input_received = True
            if key == 'left':
                monitoring_index -= 1
            elif key == 'right':
                monitoring_index += 1
            elif key >= '0' and key <= '9':
                monitoring_index = int(key) - 0
            else:
                keyboard_input_received = False
            monitoring_index = monitoring_index % len(commands)

    except e as Exception:
        print(type(e))
        print(str(e))
        stop_this_job()
        exit()


threading.Thread(target=lambda: sshkeyboard.listen_keyboard(on_press=switch_monitoring_index)) \
    .start()

finished = False
refresh_period = 5
keyboard_check_period = 0.1
while state == State.SUBMITTED:
    if not finished:
        finished, _ = print_monitoring_info(job_id, index=monitoring_index,
                                            pbs_file_path=pbs_file_name,
                                            outputs=[hline(text=pbs_file_name), pbs_file_content,
                                                     hline()],
                                            scroll_pos_fine=scroll_pos_fine,
                                            scroll_pos_pg=scroll_pos_pg)
    if finished and len(commands) < 2:
        stop_this_job()

    for i in range(int(refresh_period / keyboard_check_period + 0.5)):
        if state != State.SUBMITTED or keyboard_input_received:
            keyboard_input_received = False
            break
        time.sleep(keyboard_check_period)

stop_this_job()
os.system('clear')
print_output_file(job_id, pbs_file_name, index=monitoring_index)
