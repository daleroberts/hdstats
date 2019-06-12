import os


def get_max_threads():
    n = os.cpu_count() or 1
    print("Automatically using %i threads." % (n,))
    return n
