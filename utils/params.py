import argparse
import os


class BaseArgs:

    def __init__(self):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # hardware
        self.parser.add_argument('--n_workers', type=int, default=4, help='number of threads')
        self.parser.add_argument('--gpus', type=str, default='2', help='visible GPU ids, separated by comma')
        self.parser.add_argument('--ckpt_dir', type=str,
                                 default=os.path.join(os.getcwd(), 'ckpt'),
                                 help='the directory that contains all checkpoints')
        self.parser.add_argument('--ckpt_name', type=str, default='ckpt', help='checkpoint name')

        self.is_train, self.split = None, None

    def parse(self):
        opt = self.parser.parse_args()
        opt.is_train, opt.split = self.is_train, self.split
        log = ['Arguments: ']
        for k, v in sorted(vars(opt).items()):
            log.append('{}: {}'.format(k, v))

        return opt, log


if __name__ == "__main__":
    ins = BaseArgs()
    opt, log = ins.parse()
    print(opt)
    print(log)
