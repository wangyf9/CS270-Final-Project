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
Ori_low_datas_100 = [f for f in os.listdir(workingdir_1) if f.endswith('.mat')]
Ori_low_datas_250 = [f for f in os.listdir(workingdir_2) if f.endswith('.mat')]
Ori_low_datas_500 = [f for f in os.listdir(workingdir_3) if f.endswith('.mat')]

# Parameters
Z = 1000
epsilon = 1000000
m = 12

# Start algorithms for each file
def process_files(file_list, type, save_dir, m, epsilon, RBF_type):
    for i, file_name in enumerate(file_list):
        # Load data
        data = sio.loadmat(os.path.join(workingdir_1, file_name))
        if type == 100:
            all_data = data['data_100Hz']
        elif type == 250:
            all_data = data['data_250Hz']
        elif type == 500:
            all_data = data['data_500Hz']

        # Perform interpolation
        IQ = teulm_rbf_interpolation(all_data, m, epsilon, RBF_type, 2, save_dir)
        # Save interpolated data
        sio.savemat(os.path.join(save_dir, f'data_{type}Hz_Up_{i+1}.mat'), {'IQ': IQ})

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

def base_construct(Data, RBF_Type, epsilon):
    # data size now 78 * 118 * 80
    # choose a random row 2D layer to construct base
    selected_row_data = Data[0, :, :] # 1 * 118 * 80
    num_cols, num_frames = selected_row_data.shape
    N = num_cols * num_frames  # 118 * 80
    Phi = np.zeros((N, N), dtype=np.complex_)
    normalized_cols = np.linspace(0, 1, num_cols)
    normalized_frames = np.linspace(0, 1, num_frames)

    # generate Phi
    for i in range(num_frames):
        for j in range(num_cols):
            for m in range(num_frames):
                for n in range(num_cols):
                    idx1 = i * num_frames + j
                    idx2 = m * num_frames + n
                    r = np.linalg.norm(np.array([normalized_frames[i], normalized_cols[j]]) - np.array([normalized_frames[m], normalized_cols[n]]))
                    Phi[idx1, idx2] = RBF(r, RBF_Type, epsilon)
                    
    # Adding regularization to Phi
    reg_param = 1e-6
    Phi += reg_param * np.eye(N)
    return Phi

def ivts_construct(Data, i):
    layer_data = Data[i, :, :]
    IVTS = layer_data.flatten()
    return IVTS

def h_construction(ori_Data, new_frame_num, RBF_Type, epsilon):
    _, col, old_frame_num = ori_Data.shape
    H = np.zeros((col * new_frame_num, col * old_frame_num), dtype=np.complex_) # M = col* new_frame_num, N = col* old_frame_num
    normalized_cols = np.linspace(0, 1, col)
    normalized_frames = np.linspace(0, 1, old_frame_num)
    target_normalized_frames = np.linspace(0, 1, new_frame_num)

    for i in range(new_frame_num):
        for j in range(col):
            for m in range(old_frame_num):
                for n in range(col):
                    idx1 = i * col + j
                    idx2 = m * col + n
                    r = np.linalg.norm(np.array([target_normalized_frames[i], normalized_cols[j]]) - np.array([normalized_frames[m], normalized_cols[n]]))
                    H[idx1, idx2] = RBF(r, RBF_Type, epsilon)
    return H

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

def teulm_rbf_interpolation(D, m, epsilon, RBF_type, batch_size, save_dir):
    # D is one file of dataset 78 * 118 * 80/200/400
    # epsilon m RBF_type you can choose randomly
    # type and batch_size are corresponding 
    # 100(Hz) 80(Frame); 250 200; 500 400
    row, col, frame_num = D.shape
    sample_select_data = D[:, :, :2]
    Phi = base_construct(sample_select_data, RBF_type, epsilon)
    print(f'Size of Phi: {Phi.shape}')
    print(f'Rank of Phi: {np.linalg.matrix_rank(Phi)}')
    print(f'Cond of Phi: {np.linalg.cond(Phi)}')

    print('=== Base Construction completed!!! ===')
    # batch
    target_data = np.zeros((row, col, 800), dtype=np.complex_)
    count = 0
    total = 0
    for start_frame in range(frame_num - 1):
        end_frame = min(start_frame + 1, frame_num)
        # print(f'Start Frame: {start_frame}, End Frame: {end_frame}')
        selected_data = D[:, :, start_frame:end_frame+1]
        H = h_construction(selected_data, 11, RBF_type, epsilon)

        for i in range(row):
            IVTS_layer = ivts_construct(selected_data, i)
            Beta = np.linalg.solve(Phi, IVTS_layer)
            f_pre = H @ Beta # M * 1
            result = f_pre.reshape(col, -1)
            # print(f'Result shape: {result.shape}')
            idx = start_frame % 80
            target_data[i, :, idx*10: (idx + 1)*10] = result[:, :10]

        count += 1
        if count == 80:
            total += 1
            IQ = target_data
            sio.savemat(os.path.join(save_dir, f'data_100Hz_Up_{total}.mat'), {'IQ': IQ})
            target_data = np.zeros((row, col, 800), dtype=np.complex_)
            count = 0

# Main Execution
process_files(Ori_low_datas_100, 100, save_dir_1, m, epsilon, 'GA')