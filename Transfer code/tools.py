import sys
from time import clock


class ShowProcess():
    i = 0
    max_steps = 0
    max_arrow = 50  # length of bar
    infoDone = 'done'
    start = clock()

    def __init__(self, max_steps, infoDone='Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, i=None, start=start):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        now = int((clock() - start) / 60)
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' \
                      + '%.2f' % percent + '%    time=' + str(now) + 'min\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0