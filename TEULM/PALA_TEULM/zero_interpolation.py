import os
import numpy as np
import scipy.io as sio

# Adjust path
current_path = os.getcwd()
PALA_data_folder = os.path.join(current_path, 'TEULM/PALA_TEULM/DS_DATA')
PALA_save_folder = os.path.join(current_path, 'TEULM/PALA_TEULM/US_DATA')
os.makedirs(PALA_save_folder, exist_ok=True)

# Selected data file and saving folders
workingdir = PALA_data_folder
workingdir_1 = os.path.join(PALA_data_folder, 'DS_10_100HZ')
workingdir_2 = os.path.join(PALA_data_folder, 'DS_4_250HZ')
workingdir_3 = os.path.join(PALA_data_folder, 'DS_2_500HZ')
save_dir_1 = os.path.join(PALA_save_folder, 'US_10_100HZ')
save_dir_2 = os.path.join(PALA_save_folder, 'US_4_250HZ')
save_dir_3 = os.path.join(PALA_save_folder, 'US_2_500HZ')
os.makedirs(save_dir_1, exist_ok=True)
os.makedirs(save_dir_2, exist_ok=True)
os.makedirs(save_dir_3, exist_ok=True)

# Load Original data
Ori_low_datas_100 = [f for f in os.listdir(workingdir_1) if f.endswith('.mat')][:8]
Ori_low_datas_250 = [f for f in os.listdir(workingdir_2) if f.endswith('.mat')]
Ori_low_datas_500 = [f for f in os.listdir(workingdir_3) if f.endswith('.mat')]

# Parameters
Z = 1000
epsilon = 3
m = 12

# Zero Interpolate
def zero_interpolate(file_list, type, save_dir, m, epsilon, RBF_type):
    for i, file_name in enumerate(file_list):
        # Load data
        data = sio.loadmat(os.path.join(workingdir_1, file_name))
        if type == 100:
            single_data = data['data_100Hz']
        elif type == 250:
            single_data = data['data_250Hz']
        elif type == 500:
            single_data = data['data_500Hz']

        # Perform interpolation
        IQ = init_interpolation_set(single_data, type)
        # Save interpolated data
        sio.savemat(os.path.join(save_dir, f'data_{type}Hz_Up_{i+1}.mat'), {'IQ': IQ})

# Initialize Interpolation Set
def init_interpolation_set(ori_data, type):
    Interpolation_Set = np.zeros((78, 118, 800), dtype=np.complex_)  # Allow complex numbers
    if type == 100:
        for i in range(80):
            idx = i * 10
            Interpolation_Set[:, :, idx] = ori_data[:, :, i]
            for j in range(1, 9):
                Interpolation_Set[:, :, idx + j] = 0
    elif type == 250:
        for i in range(200):
            idx = i * 4
            Interpolation_Set[:, :, idx] = ori_data[:, :, i]
            for j in range(1, 3):
                Interpolation_Set[:, :, idx + j] = 0
    elif type == 500:
        for i in range(400):
            idx = i * 2
            Interpolation_Set[:, :, idx] = ori_data[:, :, i]
            Interpolation_Set[:, :, idx + 1] = 0
    return Interpolation_Set

# PreProcessing
def data_preprocess(ori_data, type):
    all_data = []
    for file_name in ori_data:
        data = sio.loadmat(os.path.join(workingdir_1, file_name))
        if type == 100:
            all_data.append(data['data_100Hz'])
        elif type == 250:
            all_data.append(data['data_250Hz'])
        elif type == 500:
            all_data.append(data['data_500Hz'])
    return np.concatenate(all_data, axis=2)

# Base Construction
def base_construct(Data, RBF_Type, epsilon):
    selected_row_data = Data[0, :, :]
    num_cols, num_frames = selected_row_data.shape
    N = num_cols * num_frames
    Phi = np.zeros((N, N))
    normalized_cols = np.linspace(0, 1, num_cols)
    normalized_frames = np.linspace(0, 1, num_frames)

    for i in range(num_cols):
        for j in range(num_frames):
            for m in range(num_cols):
                for n in range(num_frames):
                    idx1 = i * num_frames + j
                    idx2 = m * num_frames + n
                    r = np.linalg.norm([normalized_cols[i], normalized_frames[j]] - [normalized_cols[m], normalized_frames[n]])
                    Phi[idx1, idx2] = RBF(r, RBF_Type, epsilon)
    return Phi

# IVTS Construction
def ivts_construct(Data, i):
    layer_data = Data[i, :, :]
    IVTS = layer_data.flatten()
    return IVTS

# H Construction
def h_construction(ori_Data, new_frame_num, RBF_Type, epsilon):
    _, col, old_frame_num = ori_Data.shape
    H = np.zeros((col * new_frame_num, col * old_frame_num))
    normalized_cols = np.linspace(0, 1, col)
    normalized_frames = np.linspace(0, 1, old_frame_num)
    target_normalized_frames = np.linspace(0, 1, new_frame_num)

    for i in range(col):
        for j in range(new_frame_num):
            for m in range(col):
                for n in range(old_frame_num):
                    idx1 = i * new_frame_num + j
                    idx2 = m * old_frame_num + n
                    r = np.linalg.norm([normalized_cols[i], target_normalized_frames[j]] - [normalized_cols[m], normalized_frames[n]])
                    H[idx1, idx2] = RBF(r, RBF_Type, epsilon)
    return H

# RBF Function
def RBF(r, type, epsilon):
    if type == 'GA':
        return np.exp(-epsilon * (r ** 2))
    elif type == 'MQ':
        return np.sqrt(1 + (epsilon * r) ** 2)
    elif type == 'IMQ':
        return 1 / np.sqrt(1 + (epsilon * r) ** 2)
    elif type == 'ThinPlateSpline':
        return (r ** 2) * np.log(abs(r)) if r != 0 else 0
    elif type == 'Cubic':
        return abs(r) ** 3
    else:
        raise ValueError('Unknown RBF type. Supported types are Gaussian, Multiquadric, InverseMultiquadric, ThinPlateSpline, and Cubic.')

# Main Execution
zero_interpolate(Ori_low_datas_100, 100, save_dir_1, m, epsilon, 'GA')