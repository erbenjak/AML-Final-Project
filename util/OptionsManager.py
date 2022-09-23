import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

class OptionsManager:

    def __init__(self):
        self.isInitialized = False

    @staticmethod
    def load_default_options(parser):
        parser.add_argument("PathToData", help="the path needs to have the correct sub-folders")
        parser.add_argument("ImageStoragePath", help="the path for storing the images")
        parser.add_argument("ModelStoragePath", help="the path for storing the learning progress")
        parser.add_argument("LossStoragePath", help="the path for storing the intermediate mean losses")
        parser.add_argument("LearningRate", type=float, default=0.1, help="the used learning rate")
        parser.add_argument("DatasetName", help="the name of the dataset [Apples2Oranges, ...]")
        parser.add_argument("isTrain", type=boolean_string, default=True, help="Puts the net into training mode")
        parser.add_argument("GeneratorIsNineBlock", type=boolean_string, help="determines size of gen-networks")
        parser.add_argument('--Beta1', type=float, default=0.9, help="The first beta param of ADAM")
        parser.add_argument('--Beta2', type=float, default=0.999, help="The second beta param of ADAM")
        parser.add_argument('--isRandomCrop', type=boolean_string, default=False, help="input transform")
        parser.add_argument('--GanFactor', type=int, default=1, help="fraction of gan-loss in total loss")
        parser.add_argument('--CyclicFactor', type=int, default=10, help="fraction of cyclic loss in total loss")
        parser.add_argument('--ImageBufferSize', type=int, default=100, help="size  of the used image buffer")
        parser.add_argument('--BatchSize', type=int, default=2, help="size of the processed batches ")
        return parser

    def load_all_options(self):
        """ First the default options are loaded and secondly the commandline arguments are passed"""

        if not self.isInitialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.load_default_options(parser)

        opt, _ = parser.parse_known_args()
        return opt


