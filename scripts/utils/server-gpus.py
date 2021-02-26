import os
import sys
import subprocess
import string
import random

bashfile=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
bashfile='/tmp/'+bashfile+'.sh'

f = open(bashfile, 'w')
s = """declare -a servers=("strider.zemris.fer.hr" "celeborn" "treebeard" "magellan" "shelob" "nazgul")

for server in "${servers[@]}"; do
  echo $server $HOSTNAME
  ssh "$USER"@$server "nvidia-smi | sed '/^ *$/q'"
"""
f.write(s)
f.close()
os.chmod(bashfile, 0o755)
bashcmd=bashfile
for arg in sys.argv[1:]:
  bashcmd += ' '+arg
subprocess.call(bashcmd, shell=True)
