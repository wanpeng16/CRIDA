

def build_dataset(dataset):
    from datasets.image_dataset import DynamicCEUS_Images
    DataSet = DynamicCEUS_Images
    if dataset.endswith('rank_pooling_early'):
        from datasets.dataset import DynamicCEUS_RankPoolingEarly
        DataSet = DynamicCEUS_RankPoolingEarly
    elif dataset.endswith('lssl_early'):
        from datasets.dataset import DynamicCEUS_LSSLEarly
        DataSet = DynamicCEUS_LSSLEarly
    elif dataset.endswith('cca'):
        from datasets.dataset import DynamicCEUS_RankPoolingEarlyCCA
        DataSet = DynamicCEUS_RankPoolingEarlyCCA
    return DataSet


def build_model(model):
    from module.temporal_aggregation import ClassifierRankPoolingEarly, \
        ClassifierLSSLEarly
    from module.deep_network import ClassifierVGGBase,ClassifierVGGUSBase
    model_map = {
        'pyro_rank_pooling_early': ClassifierRankPoolingEarly,
        'pyro_lssl': ClassifierLSSLEarly,
        'single_us':ClassifierVGGUSBase,
    }

    return model_map[model] if model in model_map.keys() else ClassifierVGGBase


def build_pooling_method(method):
    from module.pooling_method import early_pooling, rank_pooling_once, average_pooling, Rank_Pooling, \
        self_attention_pooling
    pooling_map = {
        "early": early_pooling,
        "pooling_once": rank_pooling_once,
        "average_pooling": average_pooling,
        'self_attention': self_attention_pooling
    }
    return pooling_map[method] if method in pooling_map.keys() else Rank_Pooling


def build_trainer_tester(opt):
    from utils.train_deep import TrainerTesterDeep
    from utils.train import TrainerTesterExcel
    policy = opt.policy
    method_map = {
        'ISDA_cluster': TrainerTesterDeep,

    }
    if policy in method_map.keys():
        return method_map[policy]

    model_map = {

    }
    if opt.model in model_map.keys():
        return model_map[opt.model]
    if 'excel' == opt.data_type:
        return TrainerTesterExcel
    return TrainerTesterDeep
