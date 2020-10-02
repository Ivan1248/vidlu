import os
import time
from datetime import datetime
import sys


def run_command(cmd):
    with os.popen(cmd) as stream:
        return stream.read()


class OutputBuilder:
    def __init__(self):
        self.output = ''

    def add(self, text, line_numbers=None):
        self.output += text

    def filter_add(self, text, line_numbers=None):
        text = text.split('\n')
        for l in line_numbers:
            self.output += text[l] + '\n'

    def __str__(self):
        return self.output


while True:
    ob = OutputBuilder()
    ob.add(f"[{datetime.now().strftime('%H:%M:%S')}]\n")
    ob.filter_add(run_command('sensors'), [2, 3, 7])
    ob.filter_add(run_command('nvidia-smi'), [5, 8, 11, 14])
    # for x in 'abc':
    #     ob.add(run_command('hddtemp /dev/sd' + x))
    print(str(ob))
    sys.stdout.flush()
