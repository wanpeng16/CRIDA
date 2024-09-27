import os

config = {
    'running_config': 'pyro_liver',
    'seeds': range(0,5),
    'lambda0': 0.25,
    'epochs': 200,
    'pretrain_epochs': 20,
    'every_epoch': 1,
    'gpu': '2',
    'batch_size': 32,
    'hidden_dims': [128],
    'lr': 0.0003,
    'num_workers': 0,
    'policy': 'default',
    'seed': 0,
    'data_type': 'image',
    'cca_method': None,
    'latent_dimensions': None,
    'num_clusters': 3,
}

running_config = {
    'pyro_breast': {
        'model': 'default',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/pyro_breast',
    },
    'pyro_liver': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/pyro_liver',
    },
    # Sensibility
    'sen_pyro_liver_pre_10': {
        'model': 'default',
        'dataset': 'liver_images',
        'pretrain_epochs': 10,
        'ckpt_dir': 'ckpt/sen_pyro_liver_pre_10',
    },
    'sen_pyro_liver_lambda_1': {
        'model': 'default',
        'dataset': 'liver_images',
        'lambda0': 1,
        'ckpt_dir': 'ckpt/sen_pyro_liver_lambda_1',
    },
    'sen_pyro_liver_lambda_0': {
        'model': 'default',
        'dataset': 'liver_images',
        'lambda0': 0,
        'ckpt_dir': 'ckpt/sen_pyro_liver_lambda_0',
    },
    'sen_pyro_liver_pre_5': {
        'model': 'default',
        'dataset': 'liver_images',
        'pretrain_epochs': 5,
        'ckpt_dir': 'ckpt/sen_pyro_liver_pre_5',
    },
    'sen_pyro_liver_pre_0': {
        'model': 'default',
        'dataset': 'liver_images',
        'pretrain_epochs': 0,
        'ckpt_dir': 'ckpt/sen_pyro_liver_pre_0',
    },
    'sen_pyro_liver_pre_15': {
        'model': 'default',
        'dataset': 'liver_images',
        'pretrain_epochs': 15,
        'ckpt_dir': 'ckpt/sen_pyro_liver_pre_15',
    },
    'sen_pyro_liver_k_3_s_3': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_3_s_3',
        'num_clusters': 3,
        'sigma': 3,
    },
    'sen_pyro_liver_k_3_s_5': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_3_s_5',
        'num_clusters': 3,
        'sigma': 5,
    },
    'sen_pyro_liver_k_3_s_10': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_3_s_10',
        'num_clusters': 3,
        'sigma': 10,
    },
    'sen_pyro_liver_k_5_s_10': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_3_s_10',
        'num_clusters': 5,
        'sigma': 10,
    },
    'sen_pyro_liver_k_4_s_10': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_4_s_10',
        'num_clusters': 4,
        'sigma': 10,
    },
    'sen_pyro_liver_k_4_s_1': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_4_s_1',
        'num_clusters': 4,
        'sigma': 1,
    },
    'sen_pyro_liver_k_5_s_1': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_5_s_1',
        'num_clusters': 5,
        'sigma': 1,
    },
    'sen_pyro_liver_k_2_s_1': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_2_s_1',
        'num_clusters': 3,
        'sigma': 1,
    },
    'sen_pyro_liver_k_2_s_0_5': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_2_s_0_5',
        'num_clusters': 2,
        'sigma': 0.5,
    },
    'sen_pyro_liver_k_4_s_0_5': {
        'model': 'default',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/sen/sen_pyro_liver_k_4_s_0_5',
        'num_clusters': 4,
        'sigma': 0.5,
    },
    'pyro_liver_full': {
        'model': 'default',
        'dataset': 'liver_images_full',
        'ckpt_dir': 'ckpt/pyro_liver_full',
    },
    # for liver pooling method
    'pyro_liver_rank_pooling_early': {
        'model': 'pyro_rank_pooling_early',
        'dataset': 'liver_rank_pooling_early',
        'ckpt_dir': 'ckpt/pyro_liver_rank_pooling_early',
        'epochs': 50,
    },
    'pyro_lssl_early': {
        'model': 'pyro_lssl',
        'dataset': 'liver_lssl_early',
        'ckpt_dir': 'ckpt/pyro_lssl_early',
        'epochs': 100,
    },
    'pyro_liver_rank_pooling_once': {
        'model': 'pyro_rank_pooling_once',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/pyro_liver_rank_pooling_once',
        'pretrain_epochs': 10,
        'epochs': 30,
    },

    'pyro_average_pooling': {
        'model': 'pyro_average_pooling',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/pyro_average_pooling',
        'pretrain_epochs': 10,
        'epochs': 20,
    },

    'pyro_self_attention': {
        'model': 'pyro_self_attention',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/pyro_self_attention',
        'pretrain_epochs': 10,
        'epochs': 30,
    },
    # for breast pooling method
    'pyro_breast_rank_pooling_early': {
        'model': 'pyro_rank_pooling_early',
        'dataset': 'breast_rank_pooling_early',
        'ckpt_dir': 'ckpt/pyro_breast_rank_pooling_early',
        'epochs': 50,
    },
    'pyro_breast_lssl_early': {
        'model': 'pyro_lssl',
        'dataset': 'breast_lssl_early',
        'ckpt_dir': 'ckpt/pyro_breast_lssl_early',
        'epochs': 100,
    },

    'pyro_breast_rank_pooling_once': {
        'model': 'pyro_rank_pooling_once',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/pyro_breast_rank_pooling_once',
        'pretrain_epochs': 10,
        'epochs': 30,
    },
    'pyro_breast_average_pooling': {
        'model': 'pyro_average_pooling',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/pyro_breast_average_pooling',
        'pretrain_epochs': 20,
        'epochs': 30,
    },

    'pyro_breast_self_attention': {
        'model': 'pyro_self_attention',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/pyro_self_attention',
        'pretrain_epochs': 10,
        'epochs': 30,
    },

    # for liver enhancement method
    'pyro_interpolate': {
        'model': 'default',
        'dataset': 'liver_images',
        'policy': 'Interpolate',
        'probability': 0.7,
        'magnitude': 3,
        'ckpt_dir': 'ckpt/pyro_interpolate',

    },
    'pyro_extrapolate': {
        'model': 'default',
        'dataset': 'liver_images',
        'policy': 'Extrapolate',
        'probability': 0.6,
        'magnitude': 7,
        'ckpt_dir': 'ckpt/pyro_extrapolate',
    },
    'pyro_ISDA': {
        'model': 'default',
        'dataset': 'liver_images',
        'policy': 'ISDA',
        'ckpt_dir': 'ckpt/pyro_ISDA',
        'epochs': 50,

    },
    'pyro_ISDA_US': {
        'model': 'single_us',
        'dataset': 'liver_images',
        'policy': 'ISDA_US',
        'ckpt_dir': 'ckpt/pyro_ISDA',
        'epochs': 50,

    },
    'pyro_ISDA_cluster': {
        'model': 'default',
        'dataset': 'liver_images',
        'policy': 'ISDA_cluster',
        'ckpt_dir': 'ckpt/pyro_ISDA_cluster',
        'epochs': 50,

    },
    # for breast enhancement method
    'pyro_breast_interpolate': {
        'model': 'default',
        'dataset': 'breast_images',
        'policy': 'Interpolate',
        'probability': 0.7,
        'magnitude': 3,
        'ckpt_dir': 'ckpt/pyro_breast_interpolate',
        'epochs': 50,
    },
    'pyro_breast_extrapolate': {
        'model': 'default',
        'dataset': 'breast_images',
        'policy': 'Extrapolate',
        'probability': 0.6,
        'magnitude': 7,
        'ckpt_dir': 'ckpt/pyro_breast_extrapolate',
        'epochs': 50,
    },
    'pyro_breast_ISDA': {
        'model': 'default',
        'dataset': 'breast_images',
        'policy': 'ISDA',
        'ckpt_dir': 'ckpt/pyro_breast_ISDA',
        'pretrain_epochs': 10,
        'epochs': 30,
    },
    'pyro_breast_ISDA_cluster': {
        'model': 'default',
        'dataset': 'breast_images',
        'policy': 'ISDA_cluster',
        'ckpt_dir': 'ckpt/pyro_breast_ISDA_cluster',
        'pretrain_epochs': 10,
        'epochs': 30,
    },
    # for Contrast method
    'dcca_liver': {
        'model': 'cca',
        'cca_method': 'dcca',
        'dataset': 'liver_rank_pooling_early_cca',
        'ckpt_dir': 'ckpt/dcca_liver',
        'epochs': 50,
        'pretrain_epochs': 20,

    },
    'dcca_breast': {
        'model': 'cca',
        'cca_method': 'dcca',
        'dataset': 'breast_rank_pooling_early_cca',
        'ckpt_dir': 'ckpt/dcca_breast',
        'epochs': 50,
        'pretrain_epochs': 20,

    },
    'dccae_liver': {
        'model': 'cca',
        'cca_method': 'dccae',
        'dataset': 'liver_rank_pooling_early_cca',
        'ckpt_dir': 'ckpt/dccae_liver',
        'epochs': 50,
        'pretrain_epochs': 20,
        'batch_size': 128

    },

    'dccae_breast': {
        'model': 'cca',
        'cca_method': 'dccae',
        'dataset': 'breast_rank_pooling_early_cca',
        'ckpt_dir': 'ckpt/dccae_breast',
        'epochs': 50,
        'pretrain_epochs': 20,
        'batch_size': 128,

    },
    # 2015 MVCNN
    'mvcnn_liver': {
        'model': 'mvcnn',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/mvcnn_liver',
        'epochs': 50,
    },
    'mvcnn_breast': {
        'model': 'mvcnn',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/mvcnn_breast',
        'epochs': 50,
    },
    # 2024 RCML
    'rcml_liver': {
        'model': 'rcml',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/rcml_liver',
        'epochs': 50,
    },
    'rcml_breast': {
        'model': 'rcml',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/rcml_breast',
        'epochs': 50,
    },
    # 2022 TMDLO
    'tmdlo_liver': {
        'model': 'tmdlo',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/tmdlo_liver',
        'epochs': 50,
    },
    'tmdlo_breast': {
        'model': 'tmdlo',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/tmdlo_breast',
        'epochs': 50,
    },
    'tmc_liver': {
        'model': 'tmc',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/tmc_liver',
        'epochs': 50,
    },
    'tmc_breast': {
        'model': 'tmc',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/tmc_breast',

    },
    'qmf_liver': {
        'model': 'qmf',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/qmf_liver',
        'epochs': 50,
    },
    'qmf_breast': {
        'model': 'qmf',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/qmf_breast',
    },
    'mmdynamic_liver': {
        'model': 'mmdynamics',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/mmdynamic_liver',
    },
    'mmdynamic_breast': {
        'model': 'mmdynamics',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/mmdynamic_breast',
    },
    # for ablation study
    'ablation_dynamics_liver': {
        'model': 'default_without_dynamics',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/ablation_dynamics_liver',
        'epochs': 30,
        'pretrain_epochs': 10

    },
    'ablation_dynamics_breast': {
        'model': 'default_without_dynamics',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/ablation_dynamics_breast',
        'epochs': 30,
        'pretrain_epochs': 10
    },
    'ablation_enhance_liver': {
        'model': 'default',
        'dataset': 'liver_images',
        'policy': 'none',
        'ckpt_dir': 'ckpt/ablation_enhance_liver',
        'dynamics_num': 2,
        'epochs': 30,
        'pretrain_epochs': 10
    },
    'ablation_enhance_breast': {
        'model': 'default',
        'dataset': 'breast_images',
        'policy': 'none',
        'ckpt_dir': 'ckpt/ablation_enhance_breast',
        'dynamics_num': 2,
        'epochs': 30,
        'pretrain_epochs': 10
    },
    'ablation_all': {
        'model': 'default_without_dynamics',
        'policy': 'none',
        'dataset': 'liver_images',
        'ckpt_dir': 'ckpt/ablation_all',
        'epochs': 30
    },
    'ablation_all_breast': {
        'model': 'default_without_dynamics',
        'policy': 'none',
        'dataset': 'breast_images',
        'ckpt_dir': 'ckpt/ablation_all_breast',
        'pretrain_epochs': 10,
        'epochs': 30
    }
}
model_config = {
    'default': {
        'dynamics_num': 2,
        'pooling': 'rank_pooling'
    },
    'pyro_rank_pooling_early': {
        'dynamics_num': 2,
        'pooling': 'early'
    },
    'pyro_rank_pooling_once': {
        'dynamics_num': 1,
        'pooling': 'pooling_once'
    },
    'pyro_average_pooling': {
        'dynamics_num': 1,
        'pooling': 'average_pooling'
    },
    'pyro_lssl': {
        'dynamics_num': 1,
        'pooling': 'early'
    },
    'pyro_self_attention': {
        'dynamics_num': 1,
        'pooling': 'self_attention_pooling'
    },
    'cca': {
        'dynamics_num': 2,
        'pooling': 'early',
        'latent_dimensions': 128,
        'num_workers': 4,
    },
    'tmc': {
        'dynamics_num': 0,
    },
    'qmf': {
        'dynamics_num': 0,
    },
    'mmdynamics': {
        'dynamics_num': 0,
    },
    'default_without_dynamics': {
        'dynamics_num': 0,
        'pooling': 'none'
    },
    'default_without_enhance': {

    },
    'single_us': {
        'dynamics_num': 0,

    },
    'mvcnn': {
        'dynamics_num': 0,
    },
    'rcml': {
        'dynamics_num': 0,
    },
    'tmdlo': {
        'dynamics_num': 0,

    }

}
root_dir = '/home/amax/Desktop/workspace/dataset/pyro_data'
dataset_config = {
    'breast_images': {
        'lambda0': 0.25,
        'sigma': 1,
        'us_dim': 512,
        'ceus_dim': 512,
        'num_clusters': 3,
        'dset_name': 'breast',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': os.path.join(root_dir, 'breast'),
        'data_type': 'image',
        'epochs': 100,
    },

    'breast': {
        'lambda0': 0.25,
        'sigma': 1,
        'us_dim': 146,
        'ceus_dim': 152,
        'num_clusters': 3,
        'dset_name': 'breast',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': root_dir,
        'data_type': 'excel',
    },
    'liver': {
        'lambda0': 0.1,
        'sigma': 1,
        'us_dim': 145,
        'ceus_dim': 152,
        'num_clusters': 3,
        'dset_name': 'liver',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': root_dir,
        'data_type': 'excel',

    },
    'liver_images': {
        'lambda0': 0.1,
        'sigma': 1,
        'us_dim': 512,
        'ceus_dim': 512,
        'epochs': 100,
        'num_clusters': 3,
        'dset_name': 'liver',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': os.path.join(root_dir, 'liver'),
    },
    'liver_images_full': {
        'lambda0': 0.1,
        'sigma': 1,
        'us_dim': 512,
        'ceus_dim': 512,
        'epochs': 100,
        'num_clusters': 3,
        'dset_name': 'liver_full',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': os.path.join(root_dir, 'liver_full'),
    },
    'liver_rank_pooling_early': {
        'lambda0': 0.1,
        'sigma': 1,
        'us_dim': 145,
        'ceus_dim': 152,
        'num_clusters': 3,
        'dset_name': 'liver',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': root_dir,
        'data_type': 'excel',

    },
    'liver_rank_pooling_early_cca': {
        'lambda0': 0.1,
        'sigma': 1,
        'us_dim': 145,
        'ceus_dim': 152,
        'num_clusters': 3,
        'dset_name': 'liver',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': root_dir,
        'data_type': 'excel',

    },
    'breast_rank_pooling_early_cca': {
        'us_dim': 146,
        'ceus_dim': 152,
        'dset_name': 'breast',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': root_dir,
        'data_type': 'excel',
    },
    'liver_lssl_early': {
        'lambda0': 0.1,
        'sigma': 1,
        'us_dim': 145,
        'ceus_dim': 152,
        'num_clusters': 3,
        'dset_name': 'liver',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': root_dir,
        'data_type': 'excel',

    },
    'breast_rank_pooling_early': {
        'lambda0': 0.25,
        'sigma': 1,
        'us_dim': 146,
        'ceus_dim': 152,
        'num_clusters': 3,
        'dset_name': 'breast',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': root_dir,
        'data_type': 'excel',

    },
    'breast_lssl_early': {
        'lambda0': 0.25,
        'sigma': 1,
        'us_dim': 146,
        'ceus_dim': 152,
        'num_clusters': 3,
        'dset_name': 'breast',
        'num_class': 2,
        'train_type': 'tvt',
        'dset_dir': root_dir,
        'data_type': 'excel',

    }
}
config.update(running_config[config['running_config']])
config.update(dataset_config[config['dataset']])
config.update(model_config[config['model']])
config.update(running_config[config['running_config']])
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
