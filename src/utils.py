import time

def get_elapsed(start):
    return time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
