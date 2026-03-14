from datetime import datetime


class Logger:
    # TODO: replace with something better
    def __init__(self, lines=None, line_verbosities=None, printing_threshold=0, print_fn=print):
        if lines is not None and line_verbosities is None:
            line_verbosities = [1] * len(lines)
        self.lines = lines or []
        self.line_verbosities = line_verbosities or []
        self.printing_threshold = printing_threshold
        self.print_fn = print_fn

    def sublogger(self, printing_threshold=None, print_fn=None):
        return Logger(lines=self.lines, line_verbosities=self.line_verbosities,
                      printing_threshold=(self.printing_threshold if printing_threshold is None
                                          else printing_threshold),
                      print_fn=print_fn if print_fn is not None else self.print_fn)

    def log(self, text: str, verbosity=1, print_fn=None):
        time_str = datetime.now().strftime('%H:%M:%S')
        text = f"[{time_str}] {text}"
        self.lines.append(text)
        self.line_verbosities.append(verbosity)
        if self.printing_threshold <= verbosity:
            (print_fn or self.print_fn)(text)

    def print_all(self):
        for line, v in zip(self.lines, self.line_verbosities):
            if self.printing_threshold <= v:
                print(line)

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state):
        self.__dict__.update(state)
