from argparse import ArgumentParser


def get_default_args():
    parser = ArgumentParser()
    parser.backend = "ckks"
    parser.n = 4096
    parser.rolls = False
    parser.strassens = False
    parser.net = "lan"
    parser.cache = False
    parser.serialize = False
    parser.mock = False
    parser.fuzz = False
    parser.fuzz_result = False
    parser.not_secure = False
    parser.toeplitz = False
    parser.fn = ""
    return parser
