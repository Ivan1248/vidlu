from datetime import datetime


class Logger:
    def __init__(self, lines=None, line_verbosities=None, printing_threshold=0):
        if lines is not None and line_verbosities is None:
            line_verbosities = [1] * len(lines)
        self.lines = lines or []
        self.line_verbosities = line_verbosities or []
        self.printing_threshold = printing_threshold

    def sublogger(self, printing_threshold=None):
        return Logger(lines=self.lines, line_verbosities=self.line_verbosities,
                      printing_threshold=(self.printing_threshold if printing_threshold is None
                                          else printing_threshold))

    def log(self, text: str, verbosity=1):
        time_str = datetime.now().strftime('%H:%M:%S')
        text = f"[{time_str}] {text}"
        self.lines.append(text)
        self.line_verbosities.append(verbosity)
        if self.printing_threshold <= verbosity:
            print(text)

    def print_all(self):
        for line, v in zip(self.lines, self.line_verbosities):
            if self.printing_threshold <= v:
                print(line)

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state):
        self.__dict__.update(state)
