clear all;close('all')

%% Adjust path
% Define target data path
currentPath = pwd;
PALA_data_folder = [currentPath,'\DS_DATA\'];
PALA_save_folder = [currentPath,'\US_DATA\'];
if ~exist(PALA_save_folder, 'dir')
    mkdir(PALA_save_folder);
end

% Selected data file and saving folders
workingdir = [PALA_data_folder];
workingdir_1 = [PALA_data_folder 'DS_10_100HZ\'];
workingdir_2 = [PALA_data_folder 'DS_4_250HZ\'];
workingdir_3 = [PALA_data_folder 'DS_2_500HZ\'];
save_dir = [PALA_save_folder];
save_dir_1 = [save_dir '\US_10_100HZ'];mkdir(save_dir_1)
save_dir_2 = [save_dir '\US_4_250HZ'];mkdir(save_dir_2)
save_dir_3 = [save_dir '\US_2_500HZ'];mkdir(save_dir_3)
filename = 'PALA_TEULM_';
cd(workingdir)

% D Load Original data
Ori_low_datas_100 = dir([workingdir_1 '*.mat']);
Ori_low_datas_250 = dir([workingdir_2 '*.mat']);
Ori_low_datas_500 = dir([workingdir_3 '*.mat']);
size(Ori_low_datas_100)

% Start algorithms
% B target data
Z = 1000;
cd(currentPath)

% epsilon shape parameter
epsilon = 3;

% m condition value
m = 12;

% Start algorithms for each file
process_files(Ori_low_datas_100, 100, save_dir_1, m, epsilon, 'GA');
process_files(Ori_low_datas_250, 250, save_dir_2, m, epsilon, 'GA');
process_files(Ori_low_datas_500, 500, save_dir_3, m, epsilon, 'GA');

%% PROCESS
function process_files(file_list, type, save_dir, m, epsilon, RBF_type)
    for i = 1:numel(file_list)
        % Load data
        data = load([file_list(i).folder filesep file_list(i).name]);
        
        switch type
            case 100
                all_data = data.data_100Hz;
            case 250
                all_data = data.data_250Hz;
            case 500
                all_data = data.data_500Hz;
        end
        
        % Perform interpolation
        IQ = teulm_rbf_interpolation(all_data, m, epsilon, RBF_type);
        % Save interpolated data
        save([save_dir filesep 'data_' num2str(type) 'Hz_Up_' num2str(i) '.mat'], 'IQ');
    end
end

%% Base Construction
function Phi = base_construct(Data, RBF_Type, epsilon)
   % data size now 78 * 118 * 80
   % choose a random row 2D layer to construct base
   selected_row_data = Data(1, :, :);     % 1 * 118 * 80
   selected_row_data = squeeze(selected_row_data);     % 118 * 80 
   [num_cols, num_frames] = size(selected_row_data);   % C = 118, F = 80
   N = num_cols * num_frames;                          % N = 118 * 80
   Phi = zeros(N, N);
   size(Phi)
   normalized_cols = linspace(0, 1, num_cols);
   normalized_frames = linspace(0, 1, num_frames);

   % generate Phi
   for i = 1:num_cols
        for j = 1:num_frames
            for m = 1:num_cols
                for n = 1:num_frames
                    idx1 = (i-1) * num_frames + j;
                    idx2 = (m-1) * num_frames + n;
                    r = norm([normalized_cols(i), normalized_frames(j)] - [normalized_cols(m), normalized_frames(n)]);
                    Phi(idx1, idx2) = RBF(r, RBF_Type, epsilon);
                end
            end
        end
   end
end

%% F(IVTS) Construction
function IVTS = ivts_construct(Data, i)
    % extract
    layer_data = squeeze(Data(i, :, :));  
    % get 1D column vector
    IVTS = layer_data(:);  
    size(IVTS)
end

%% H Construction
function H = h_construction(ori_Data, target_data, RBF_Type, epsilon)
    [~, ~, old_frame_num] = size(ori_Data);                     
    [~, col, new_frame_num] = size(target_data);

    H = zeros(col* new_frame_num, col* old_frame_num);  % M = col* new_frame_num, N = col* old_frame_num
    % Normalized scale
    normalized_cols = linspace(0, 1, col);
    normalized_frames = linspace(0, 1, old_frame_num);
    target_normalized_frames = linspace(0, 1, new_frame_num);
    % get H
    for i = 1: col
        for j = 1: new_frame_num % M
            for m = 1: col
                for n = 1: old_frame_num % N
                    idx1 = (i-1) * new_frame_num + j;
                    idx2 = (m-1) * old_frame_num + n;
                    r = norm([normalized_cols(i), target_normalized_frames(j)] - [normalized_cols(m), normalized_frames(n)]);
                    H(idx1, idx2) = RBF(r, RBF_Type, epsilon);
                end
            end
        end
    end
end

%% UPS 2D Interpolation
function target_data = teulm_rbf_interpolation(D, m, epsilon, RBF_type)
    % D is one file of dataset 78 * 118 * 80/200/400
    % epsilon m RBF_type you can choose randomly
    % type and batch_size are corresponding 
    % 100(Hz) 80(Frame); 250 200; 500 400
    
    [row, col, frame_num] = size(D);              % row = 78 col = 118
    Phi = base_construct(D, RBF_type, epsilon);
    Phi = (Phi + 0.01* eye(size(Phi)));
    disp('=== Base Construction completed!!! ===');
    % batch 
    target_data = zeros(row, col, 800);
    H  = h_construction(D, target_data, RBF_type, epsilon);
    size(H)
    disp('=== H Construction  completed!!! ===');
    for i = 1: row
        IVTS_layer = ivts_construct(D, i);
        Beta = IVTS_layer \ Phi;
        size(Beta)
        f_pre = H * Beta';               % M * 1
        result = reshape(f_pre, col, 800); % M = 118 * 800 col * new frame num
        target_data(i, :, :) = result;
        disp(['===  Interpolation completed for layer ' num2str(i) '!!! ===']);
    end
end

%% RBF Funtion
function kernel_value = RBF(r, type, epsilon)
    switch type
        case 'GA'
            kernel_value = exp( - epsilon * (r)^ 2);
        case 'MQ'
            kernel_value = sqrt(1 + (epsilon * r)^ 2);
        case 'IMQ'
            kernel_value = 1 / sqrt(1 + (epsilon * r)^ 2);
        case 'ThinPlateSpline'
            if r == 0
                kernel_value = 0;
            else
                kernel_value = (r^2) * log(abs(r));
            end
        case 'Cubic'
            kernel_value = abs(r)^3;
        otherwise
            error('Unknown RBF type. Supported types are Gaussian, Multiquadric, InverseMultiquadric, ThinPlateSpline, and Cubic.');
    end
end



