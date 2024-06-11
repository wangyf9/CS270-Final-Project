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

% Load Original data
Ori_low_datas_100 = dir([workingdir_1 '*.mat']);
Ori_low_datas_250 = dir([workingdir_2 '*.mat']);
Ori_low_datas_500 = dir([workingdir_3 '*.mat']);
size(Ori_low_datas_100)
% Start algorithms

% D original data
all_100_data = data_preprocess(Ori_low_datas_100, 100);
[num_rows, num_cols, total_frames_100] = size(all_100_data);        %78 * 118 * 19200 actually three dimensions
all_250_data = data_preprocess(Ori_low_datas_250, 250);
[num_rows, num_cols, total_frames_250] = size(all_250_data);         %78 * 118 * 19200 * 2.5 actually three dimensions
all_500_data = data_preprocess(Ori_low_datas_500, 500);
[num_rows, num_cols, total_frames_500] = size(all_500_data);         %78 * 118 * 19200 * 5 actually three dimensions

% B target data
Z = 1000;
target_B = zeros(num_rows, num_cols, Z);
cd(currentPath)

% epsilon shape parameter
epsilon = 3;

% m condition value
m = 12;

Final_Data = teulm_rbf_interpolation(all_100_data, m, epsilon, 'GA', 100);
size(Final_Data)
block_size = 800;
num_blocks = 240;
start_indices = 1:block_size:size(interpolated_data, 3);
end_indices = min(start_indices + block_size - 1, size(interpolated_data, 3));
for i = 1:num_blocks
    block_data = interpolated_data(:, :, start_indices(i):end_indices(i));
    save([save_dir_1 filesep 'data_100Hz_Up_' num2str(i) '.mat'], 'IQ');
end

%
%% PreProcessing
function all_data = data_preprocess(ori_data, type)
    % get paras
    Nbuffers = numel(ori_data);
    all_data = [];
    for i = 1:Nbuffers
        % load data
        if type == 100
            load([ori_data(i).folder filesep ori_data(i).name], 'data_100Hz');
            all_data = cat(3, all_data, data_100Hz);
        end
        if type == 250
            load([ori_data(i).folder filesep ori_data(i).name], 'data_250Hz');
            all_data = cat(3, all_data, data_250Hz);
        end
        if type == 500
            load([ori_data(i).folder filesep ori_data(i).name], 'data_500Hz');
            all_data = cat(3, all_data, data_500Hz);
        end
    end
end

%% Interpolation Set Init
function Interpolation_Set = init_interpolation_set(ori_data, type)
    Interpolation_Set = zeros(78, 118, 192000);
    % 100Hz interpolate 9 frame per frame
    if type == 100
        for i = 1: 19200
            idx = (i - 1)* 10 + 1;
            Interpolation_Set(:,:, idx) = ori_data(:,:, i);
            % interpolate blank frame
            for j = 1: 9
                Interpolation_Set(:,:, idx + j) = 0;
            end
        end
    end
    % 250Hz interpolate 3 frame per frame
    if type == 250
        for i = 1: 48000
            idx = (i - 1)* 4 + 1;
            Interpolation_Set(:,:, idx) = ori_data(:,:, i);
            % interpolate blank frame
            for j = 1: 3
                Interpolation_Set(:,:, idx + j) = 0;
            end
        end
    end
    % 500Hz interpolate 1 frame per frame
    if type == 500
        for i = 1: 96000
            idx = (i - 1)* 2 + 1;
            Interpolation_Set(:,:, idx) = ori_data(:,:, i);
            % interpolate blank frame
            Interpolation_Set(:,:, idx + j) = 0;
        end
    end
end

%% Base Construction
function Phi = base_construct(Data, RBF_Type, epsilon)
   % choose a random row 2D layer to construct base
   selected_row_data = Data(1, :, :);     % 1 * 118 * 192000
   selected_row_data = squeeze(selected_row_data);     % 118 * 192000 
   [num_cols, num_frames] = size(selected_row_data);   % C = 118, F = 192000
   N = num_cols * num_frames;                          % N = 118 * 192000
   Phi = zeros(N, N);
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
end

%% H Construction
function [H, interpolate_Data] = h_construction(ori_Data, type, RBF_Type, epsilon)
    [~, ~, old_frame_num] = size(ori_Data);
    interpolate_Data = init_interpolation_set(ori_Data, type);
    [~, col, new_frame_num] = size(interpolate_Data);
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
function interpolate_Data = teulm_rbf_interpolation(D, m, epsilon, RBF_type, type)
    % D
    % epsilon 
    [row, ~, ~] = size(D);              % row = 78
    Phi = base_construct(D, RBF_type, epsilon);
    disp('=== Base Construction completed!!! ===');
    [H, interpolate_Data] = h_construction(D, type, RBF_type, epsilon);
    disp('=== H Construction  completed!!! ===');
    for i = 1: row
        IVTS_layer = ivts_construct(D, i);
        Beta = Phi \ IVTS_layer;
        f_pre = H * Beta;
        interpolate_Data(i,:,:) = f_pre;
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



