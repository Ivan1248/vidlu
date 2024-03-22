#!/bin/bash

# Usage: [QUEUE=<number>] [SELECT=<number>] [NGPUS=<number>] [NCPUS=<number>] bash mqsub.sh <command> [arguments...]
#
# Enqueues a job with qsub and displays its output for monitoring. If the SELECT, NGPUS, and NCPUS environment variables are not provided, the script defaults to SELECT=1 (a single node), NGPUS=1 (1 GPU) and NCPUS=32 (32 CPUs). The job runs until completed or stopped by sending SIGINT or SIGTERM.

# Usage examples:
# bash qsub.sh run-singlegpu.sh ~/path/to/script.py --arg1 value1 --arg2 value2
#   - Enqueues a job with the default configuration (SELECT=1, NGPUS=1, NCPUS=48).
# NGPUS=4 NCPUS=64 bash qsub.sh torchrun-singlenode.sh ~/path/to/script.py --arg1 value1 --arg2 value2
#   - Enqueues a job with 4 GPUs and 64 CPUs.
# SELECT=2 NGPUS=4 NCPUS=64 bash qsub.sh torchrun-singlenode.sh ~/path/to/script.py --arg1 value1 --arg2 value2
#   - Enqueues a job with 4 GPUs and 64 CPUs per node on 2 nodes.
#
# Note:
# - Consider increasing the number of CPUs if data loading is slow.

if [[ $1 == "-h" || $1 == "--help" ]]; then
    echo "Usage: bash qsub.sh <command> [arguments...]"
    echo "Enqueues a job using qsub and displays its output. The job runs until completed or stopped by sending SIGINT or SIGTERM."
    echo "Options:"
    echo "  -h, --help     Print this help message"
    exit 0
fi


# Add removed quotes around arguments so that they can be forwarded to qsub
command=$(awk -v q="'" '
  function shellquote(s) {
    gsub(q, q "\\" q q, s)
    return q s q
  }
  BEGIN {
    for (i = 1; i < ARGC; i++) {
      printf "%s", sep shellquote(ARGV[i])
      sep = " "
    }
    printf "\n"
  }' "$@")

# Temporary PBS file
pbs_file="run.pbs"
cat <<EOF >"$pbs_file"
#!/bin/bash

#PBS -q ${QUEUE:-"gpu"}
#PBS -l select=${SELECT:-"1"}:ngpus=${NGPUS:-"1"}:ncpus=${NCPUS:-"32"}

module load scientific/pytorch/1.14.0-ngc
module load cray-pals  # multinode

cd \${PBS_O_WORKDIR:-""}
echo "Running command:" "$command"
$command
EOF

echo "Enqueueing command:"
echo "$command"
output=$(qsub "$pbs_file")
echo $output
job_id=$(echo "$output" | awk -F'.' '{print $1}')
echo $job_id

# Deletes the job, removes the PBS file. Handles the first SIGINT/SIGTERM.
end_job() {
  qdel $job_id
  rm $pbs_file
}

# Prints the whole output once the output file has been created
finish() {
  # Exit on the second SIGINT/SIGTERM
  trap - SIGINT SIGTERM
  trap "echo Exited!; exit;" SIGINT SIGTERM

  stdout_file=$pbs_file.o$job_id
  echo "Waiting for the output file, $stdout_file..."
  until [ -e "$stdout_file" ]; do
    sleep 1
  done
  sleep 1

  echo "Output ('cat $stdout_file'):"
  cat $stdout_file
  stderr_file=$pbs_file.e$job_id
  echo "Run 'cat $stderr_file' for printing the error output."

  exit
}
#trap "end_job; exit;" SIGINT SIGTERM

# In a subshell so that the main shell is not interrupted by the signals
(
  trap 'end_job; exit;' SIGINT SIGTERM
  watch -n 5 "
  printf 'qstat -sw $job_id:\n'; qstat -sw $job_id | tail -n 2;
  echo
  printf '=%.0s' {1..79}
  # printf 'qtail $job_id\n'; qtail $job_id
  printf '\nqcat $job_id\n'; qcat $job_id | tail -f -n 25;
  echo
  printf '=%.0s' {1..79}
  printf '\nqcat -e $job_id\n'; qcat -e $job_id | tail -f -n 1;
  echo Press Ctrl+C to close and print the output."
)

end_job
finish
