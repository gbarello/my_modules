import numpy as np
import pickle


def dump_file(name,data):
    F = open(name,"wb")
    pickle.dump(data,F)
    F.close()
    
def fetch_file(name):
    F = open(name,"rb")
    out = pickle.load(F)
    F.close()
    return out

def csv_line(value_parser):
    """
    Return a function that parses a line of comma separated values.

    Examples:

    >>> csv_line(int)('1, 2, 3')
    [1, 2, 3]
    >>> csv_line(float)('0.5, 1.5')
    [0.5, 1.5]

    For example, it can be passed to type argument of
    `argparse.ArgumentParser.add_argument` as follows::

        parser.add_argument(
            ...,
            type=csv_line(float),
            help='Comma separated value of floats')

    """
    def convert(string):
        return list(map(value_parser, string.split(',')))
    return convert

