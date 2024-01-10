from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/zgt/CVPR2024/OSTrack-main/data/got10k_lmdb'
    settings.got10k_path = '/home/local_data/zgt/CVPR2024/OSTrack-main/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/zgt/CVPR2024/OSTrack-main-diffusion/data/itb'
    settings.lasot_extension_subset_path = '/home/data/hxt/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/zgt/CVPR2024/OSTrack-main-diffusion/data/lasot_lmdb'
    settings.lasot_path = '/home/data/hxt/data/lasot'
    settings.network_path = '/home/data/zgt/CVPR2024/OSTrack-main-diffusion/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home//zgt/CVPR2024/OSTrack-main-diffusion/data/nfs'
    settings.otb_path = '/home/data/zgt/data/OTB_sentences'
    settings.prj_dir = '/home/data/zgt/CVPR2024/OSTrack-main-diffusion'
    settings.result_plot_path = '/home/data/zgt/CVPR2024/OSTrack-main-diffusion/output/test/result_plots'
    settings.results_path = '/home/data/zgt/CVPR2024/OSTrack-main-diffusion/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/data/zgt/CVPR2024/OSTrack-main-diffusion/output'
    settings.segmentation_path = '/home/zgt/CVPR2024/OSTrack-main-diffusion/output/test/segmentation_results'
    settings.tc128_path = '/home/local_data/zgt/CVPR2024/OSTrack-main-diffusion/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/data/zgt/data/TNL2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/local_data/zgt/CVPR2024/OSTrack-main/data/trackingnet'
    settings.uav_path = '/home/local_data/zgt/CVPR2024/OSTrack-main/data/uav'
    settings.vot18_path = '/home/local_data/zgt/CVPR2024/OSTrack-main/data/vot2018'
    settings.vot22_path = '/home/local_data/zgt/CVPR2024/OSTrack-main/data/vot2022'
    settings.vot_path = '/home/local_data/zgt/CVPR2024/OSTrack-main/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

