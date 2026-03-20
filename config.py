dataset_config = {
    'trento': {
        'dataset_dir': './DATASET/trento',
        'num_class': 6,
        'hsi_n_feature': 63,
        'lidar_n_feature': 1,
        'concate_pixel': (3, 3, 3, 3),
        'class_labels' : {
            -1: 'Undefined',
            1: 'Apples',
            2: 'Buildings',
            3: 'Ground',
            4: 'Woods',
            5: 'Vineyard',
            6: 'Road',
        }
    },
    'houston13': {
        'dataset_dir': './DATASET/newhuston',
        'num_class': 15,
        'hsi_n_feature': 144,  # Houston2013 原始 144 波段
        'lidar_n_feature': 1,
        'concate_pixel': (3, 3, 3, 3),
        'class_labels': {
            0: 'Unclassified',
            1: 'Healthy grass',
            2: 'Stressed grass',
            3: 'Synthetic grass',
            4: 'Trees',
            5: 'Soil',
            6: 'Water',
            7: 'Residential',
            8: 'Commercial',
            9: 'Road',
            10: 'Highway',
            11: 'Railway',
            12: 'Parking Lot 1',
            13: 'Parking Lot 2',
            14: 'Tennis Court',
            15: 'Running Track'
        },
    },
    'muufl': {
        'dataset_dir': './DATASET/muufl',
        'num_class': 11,
        'hsi_n_feature': 64,
        'lidar_n_feature': 2,
        'concate_pixel': (3, 3, 3, 3),
        'class_labels': {
            -1: 'Unlabeled',
            1: 'Trees',
            2: 'Mostly Grass',
            3: 'Mixed Ground',
            4: 'Dirt and Sand',
            5: 'Road',
            6: 'Water',
            7: 'Buildings',
            8: 'Shadow of Buildings',
            9: 'Sidewalk',
            10: 'Yellow Curb',
            11: 'Cloth Panels'
        },
    },
    'augsburg': {
        'dataset_dir': './DATASET/augsburg',
        'num_class': 11,
        'hsi_n_feature': 180,
        'lidar_n_feature': 1,
        'concate_pixel': (3, 3, 3, 3),
        'class_labels': {
            0: "Unclassified",
            1: "Forest",
            2: "Residential Area",
            3: "Industrial Area",
            4: "Low Plants",
            5: "Allotment",
            6: "Commercial Area",
            7: "Water"
        },
    },
}

custom_colors = [
    [255, 0, 0],        # 鲜红色 - Apples
    [0, 255, 0],        # 鲜绿色 - Buildings
    [0, 0, 255],        # 鲜蓝色 - Ground
    [255, 255, 0],      # 亮黄色 - Woods
    [255, 0, 255],      # 品红色 - Vineyard
    [0, 255, 255],      # 青色 - Road
    [255, 128, 0],      # 橙色 - 额外类别
    [128, 0, 255],      # 紫色 - 额外类别
    [255, 0, 128],      # 粉红色 - 额外类别
    [0, 255, 128],      # 春绿色 - 额外类别
    [128, 255, 0],      # 黄绿色 - 额外类别
    [255, 128, 128],    # 浅红色 - 额外类别
    [128, 255, 128],    # 浅绿色 - 额外类别
    [128, 128, 255],    # 浅蓝色 - 额外类别
    [255, 255, 128],    # 浅黄色 - 额外类别
]