import os
import time
from datetime import datetime


def run_command(cmd):
    with os.popen(cmd) as stream:
        return stream.read()


class OutputBuilder:
    def __init__(self):
        self.output = ''

    def add(self, text):
        self.output += text

    def filter_add(self, text, line_numbers=None):
        text = text.split('\n')
        if not line_numbers:
            line_numbers = list(range(len(text)))
        for l in line_numbers:
            self.output += text[l] + '\n'

    def print(self):
        print(self.output, flush=True)


while True:
    ob = OutputBuilder()
    ob.add(f"[{datetime.now().strftime('%H:%M:%S')}]\n")
    ob.filter_add(run_command('sensors'), None)
    ob.filter_add(run_command('nvidia-smi'), [5, 8, 11])
    for x in 'abc':
        ob.add(run_command('hddtemp /dev/sd' + x))
    ob.print()
    time.sleep(10)
