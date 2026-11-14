llada_path = "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/LLaDA-Uni-Sign/pretrained_weight/LLaDA-8B-Instruct"

# label paths
train_ds_path = {
        "CSL_News": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/train_ready_news2.json",
        "CSL_Daily": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/train_ready2.json",
        "How2Sign": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/train_how_ready.json",
        "OpenASL": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/train_asl_ready.json",
        "YoutubeASL": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/train_ytasl_ready.json"
    }

dev_ds_path = {
        "CSL_News": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/train_ready_news2.json",
        "CSL_Daily": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/dev_ready.json",
        "How2Sign": "",
        "OpenASL": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/dev_asl_ready.json",
        "YoutubeASL": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/train_ytasl_ready.json"
    }

test_ds_path = {
        "CSL_News": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/train_ready_news2.json",
        "CSL_Daily": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/test_ready.json",
        "How2Sign": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/test_how_ready.json",
        "OpenASL": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/test_asl_ready.json",
        "YoutubeASL": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/Uni-Sign/test_exp/result/gloss_info/train_ytasl_ready.json"
    }


# # video paths
# rgb_dirs = {
#             "CSL_News": './dataset/CSL_News/rgb_format',
#             "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
#             "WLASL": "./dataset/WLASL/rgb_format",
#             "How2Sign": "/inspire/ssd/project/sais-mtm/public/qlz/data/How2Sign",
#             "OpenASL": "./dataset/OpenASL/rgb_format",
#             }

# pose paths
pose_dirs = {
            "CSL_News": './dataset/CSL_News/pose_format',
            "CSL_Daily": '/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/LLaDA-Uni-Sign/dataset/CSL_Daily/pose-rtmpose-192-clean',
            "WLASL": "./dataset/WLASL/pose_format",
            "How2Sign": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/LLaDA-Uni-Sign/dataset/How2Sign_extend/{}-pose-rtmpose-192",
            "OpenASL": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/LLaDA-Uni-Sign/dataset/OpenASL/pose-rtmpose-192",
            "YoutubeASL": "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/zujinyi/Code/LLaDA-Uni-Sign/dataset/YoutubeASL/pose-rtmpose"
}