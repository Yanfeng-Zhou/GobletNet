import numpy as np
import torchio as tio

def dataset_cfg(dataet_name):

    config = {
        'CREMI':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'SIZE': (128, 128),
                'MEAN': [0.503902],
                'STD': [0.110739],
                'MEAN_H_0.0_db2': [0.505787],
                'STD_H_0.0_db2': [0.115504],
                'MEAN_H_0.1_db2': [0.515329],
                'STD_H_0.1_db2': [0.118728],
                'MEAN_H_0.2_db2': [0.524283],
                'STD_H_0.2_db2': [0.123539],
                'MEAN_H_0.3_db2': [0.532277],
                'STD_H_0.3_db2': [0.129314],
                'MEAN_H_0.4_db2': [0.539361],
                'STD_H_0.4_db2': [0.135507],
                'MEAN_H_0.5_db2': [0.545674],
                'STD_H_0.5_db2': [0.141783],
                'MEAN_H_0.3_haar': [0.534493],
                'STD_H_0.3_haar': [0.130047],
                'MEAN_H_0.3_bior1.5': [0.534219],
                'STD_H_0.3_bior1.5': [0.128970],
                'MEAN_H_0.3_bior2.4': [0.504052],
                'STD_H_0.3_bior2.4': [0.114659],
                'MEAN_H_0.3_coif1': [0.511274],
                'STD_H_0.3_coif1': [0.116867],
                'MEAN_H_0.3_dmey': [0.505451],
                'STD_H_0.3_dmey': [0.116982],
                'MEAN_LL': [0.614682],
                'STD_LL': [0.221123],
                'MEAN_LH': [0.502604],
                'STD_LH': [0.114562],
                'MEAN_HL': [0.498184],
                'STD_HL': [0.113300],
                'MEAN_HH': [0.498554],
                'STD_HH': [0.105404],
                'PALETTE': list(np.array([
                    [255, 255, 255],
                    [0, 0, 0],
                ]).flatten())
            },
        'SNEMI3D':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'SIZE': (128, 128),
                'MEAN': [0.496661],
                'STD': [0.189456],
                'MEAN_H_0.0_db2': [0.510289],
                'STD_H_0.0_db2': [0.107591],
                'MEAN_H_0.1_db2': [0.517311],
                'STD_H_0.1_db2': [0.109279],
                'MEAN_H_0.2_db2': [0.523877],
                'STD_H_0.2_db2': [0.112444],
                'MEAN_H_0.3_db2': [0.530329],
                'STD_H_0.3_db2': [0.116392],
                'MEAN_H_0.4_db2': [0.536497],
                'STD_H_0.4_db2': [0.120649],
                'MEAN_H_0.5_db2': [0.542471],
                'STD_H_0.5_db2': [0.124972],
                'MEAN_H_0.2_haar': [0.509991],
                'STD_H_0.2_haar': [0.108379],
                'MEAN_H_0.2_bior1.5': [0.508703],
                'STD_H_0.2_bior1.5': [0.106922],
                'MEAN_H_0.2_bior2.4': [0.497987],
                'STD_H_0.2_bior2.4': [0.115372],
                'MEAN_H_0.2_coif1': [0.497637],
                'STD_H_0.2_coif1': [0.113934],
                'MEAN_H_0.2_dmey': [0.520494],
                'STD_H_0.2_dmey': [0.118463],
                'MEAN_LL': [0.603247],
                'STD_LL': [0.194900],
                'MEAN_LH': [0.493574],
                'STD_LH': [0.095770],
                'MEAN_HL': [0.458264],
                'STD_HL': [0.093275],
                'MEAN_HH': [0.498709],
                'STD_HH': [0.106967],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'EPFL':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'SIZE': (256, 256),
                'MEAN': [0.550909],
                'STD': [0.119196],
                'MEAN_H_0.0_db2': [0.501591],
                'STD_H_0.0_db2': [0.110625],
                'MEAN_H_0.1_db2': [0.508539],
                'STD_H_0.1_db2': [0.111868],
                'MEAN_H_0.2_db2': [0.513748],
                'STD_H_0.2_db2': [0.114284],
                'MEAN_H_0.3_db2': [0.517631],
                'STD_H_0.3_db2': [0.118099],
                'MEAN_H_0.4_db2': [0.520415],
                'STD_H_0.4_db2': [0.122720],
                'MEAN_H_0.5_db2': [0.522673],
                'STD_H_0.5_db2': [0.127931],
                'MEAN_H_0.3_haar': [0.520436],
                'STD_H_0.3_haar': [0.119415],
                'MEAN_H_0.3_bior1.5': [0.520191],
                'STD_H_0.3_bior1.5': [0.119014],
                'MEAN_H_0.3_bior2.4': [0.510275],
                'STD_H_0.3_bior2.4': [0.114861],
                'MEAN_H_0.3_coif1': [0.508456],
                'STD_H_0.3_coif1': [0.114656],
                'MEAN_H_0.3_dmey': [0.512606],
                'STD_H_0.3_dmey': [0.115322],
                'MEAN_LL': [0.564221],
                'STD_LL': [0.196058],
                'MEAN_LH': [0.498375],
                'STD_LH': [0.098199],
                'MEAN_HL': [0.500013],
                'STD_HL': [0.103313],
                'MEAN_HH': [0.498386],
                'STD_HH': [0.111347],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [0, 0, 255],
                ]).flatten())
            },
        'UroCell':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 4,
                'SIZE': (128, 128),
                'MEAN': [0.780025],
                'STD': [0.116556],
                'MEAN_H_0.0_db2': [0.524444],
                'STD_H_0.0_db2': [0.110518],
                'MEAN_H_0.1_db2': [0.538896],
                'STD_H_0.1_db2': [0.111077],
                'MEAN_H_0.2_db2': [0.553973],
                'STD_H_0.2_db2': [0.112203],
                'MEAN_H_0.3_db2': [0.568824],
                'STD_H_0.3_db2': [0.113740],
                'MEAN_H_0.4_db2': [0.582786],
                'STD_H_0.4_db2': [0.115483],
                'MEAN_H_0.5_db2': [0.595442],
                'STD_H_0.5_db2': [0.117495],
                'MEAN_H_0.0_haar': [0.516382],
                'STD_H_0.0_haar': [0.096641],
                'MEAN_H_0.0_bior1.5': [0.516391],
                'STD_H_0.0_bior1.5': [0.095723],
                'MEAN_H_0.0_bior2.4': [0.458845],
                'STD_H_0.0_bior2.4': [0.110316],
                'MEAN_H_0.0_coif1': [0.458496],
                'STD_H_0.0_coif1': [0.110652],
                'MEAN_H_0.0_dmey': [0.486581],
                'STD_H_0.0_dmey': [0.107864],
                'MEAN_LL': [0.734153],
                'STD_LL': [0.161995],
                'MEAN_LH': [0.508020],
                'STD_LH': [0.090761],
                'MEAN_HL': [0.505250],
                'STD_HL': [0.086142],
                'MEAN_HH': [0.478739],
                'STD_HH': [0.105861],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [0, 255, 255],
                    [255, 97, 0],
                    [0, 0, 255],

                ]).flatten())
            },
        'Nanowire':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'SIZE': (256, 256),
                'MEAN': [0.492478],
                'STD': [0.163114],
                'MEAN_H_0.0_db2': [0.499102],
                'STD_H_0.0_db2': [0.119998],
                'MEAN_H_0.1_db2': [0.500022],
                'STD_H_0.1_db2': [0.120698],
                'MEAN_H_0.2_db2': [0.500640],
                'STD_H_0.2_db2': [0.121806],
                'MEAN_H_0.3_db2': [0.501329],
                'STD_H_0.3_db2': [0.122712],
                'MEAN_H_0.4_db2': [0.502835],
                'STD_H_0.4_db2': [0.123382],
                'MEAN_H_0.5_db2': [0.505229],
                'STD_H_0.5_db2': [0.123837],
                'MEAN_H_0.5_haar': [0.493535],
                'STD_H_0.5_haar': [0.138590],
                'MEAN_H_0.5_bior1.5': [0.494959],
                'STD_H_0.5_bior1.5': [0.136488],
                'MEAN_H_0.5_bior2.4': [0.500009],
                'STD_H_0.5_bior2.4': [0.121748],
                'MEAN_H_0.5_coif1': [0.501262],
                'STD_H_0.5_coif1': [0.121685],
                'MEAN_H_0.5_dmey': [0.498611],
                'STD_H_0.5_dmey': [0.121943],
                'MEAN_LL': [0.524653],
                'STD_LL': [0.137022],
                'MEAN_LH': [0.492131],
                'STD_LH': [0.120757],
                'MEAN_HL': [0.494835],
                'STD_HL': [0.120622],
                'MEAN_HH': [0.497549],
                'STD_HH': [0.119338],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'BetaSeg':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 5,
                'SIZE': (128, 128),
                'MEAN': [0.487456],
                'STD': [0.266161],
                'MEAN_H_0.0_db2': [0.505809],
                'STD_H_0.0_db2': [0.103684],
                'MEAN_H_0.1_db2': [0.518222],
                'STD_H_0.1_db2': [0.110667],
                'MEAN_H_0.2_db2': [0.529424],
                'STD_H_0.2_db2': [0.119461],
                'MEAN_H_0.3_db2': [0.538739],
                'STD_H_0.3_db2': [0.128305],
                'MEAN_H_0.4_db2': [0.545695],
                'STD_H_0.4_db2': [0.136287],
                'MEAN_H_0.5_db2': [0.550393],
                'STD_H_0.5_db2': [0.143033],
                'MEAN_H_0.4_haar': [0.546990],
                'STD_H_0.4_haar': [0.159750],
                'MEAN_H_0.4_bior1.5': [0.545414],
                'STD_H_0.4_bior1.5': [0.148578],
                'MEAN_H_0.4_bior2.4': [0.537319],
                'STD_H_0.4_bior2.4': [0.121998],
                'MEAN_H_0.4_coif1': [0.541033],
                'STD_H_0.4_coif1': [0.131447],
                'MEAN_H_0.4_dmey': [0.538356],
                'STD_H_0.4_dmey': [0.127115],
                'MEAN_LL': [0.551324],
                'STD_LL': [0.250177],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [0, 0, 255],
                    [0, 255, 0],
                    [255, 0, 255],
                    [255, 255, 0],
                ]).flatten())
            },
        'MitoEM':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'SIZE': (256, 256),
                'MEAN': [0.535179],
                'STD': [0.224291],
                'MEAN_H_0.0_db2': [0.492742],
                'STD_H_0.0_db2': [0.103812],
                'MEAN_H_0.1_db2': [0.495451],
                'STD_H_0.1_db2': [0.107968],
                'MEAN_H_0.2_db2': [0.500071],
                'STD_H_0.2_db2': [0.113500],
                'MEAN_H_0.3_db2': [0.506310],
                'STD_H_0.3_db2': [0.119461],
                'MEAN_H_0.4_db2': [0.512783],
                'STD_H_0.4_db2': [0.125504],
                'MEAN_H_0.5_db2': [0.519021],
                'STD_H_0.5_db2': [0.131396],
                'MEAN_H_0.4_haar': [0.505615],
                'STD_H_0.4_haar': [0.141312],
                'MEAN_H_0.4_bior1.5': [0.504109],
                'STD_H_0.4_bior1.5': [0.137492],
                'MEAN_H_0.4_bior2.4': [0.498121],
                'STD_H_0.4_bior2.4': [0.117916],
                'MEAN_H_0.4_coif1': [0.503294],
                'STD_H_0.4_coif1': [0.119514],
                'MEAN_H_0.4_dmey': [0.494881],
                'STD_H_0.4_dmey': [0.124719],
                'MEAN_LL': [0.561993],
                'STD_LL': [0.217496],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [0, 0, 255],
                ]).flatten())
            },
    }

    return config[dataet_name]