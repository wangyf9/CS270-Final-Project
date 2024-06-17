clear all;close('all')

%% Adjust path
% Define target data path
currentPath = pwd;
PALA_data_folder = [currentPath,'\DS_DATA\'];
PALA_save_folder = [currentPath,'\US_DATA\'];
zero_save_folder = [currentPath,'\zero_US_DATA\'];
if ~exist(PALA_save_folder, 'dir')
    mkdir(PALA_save_folder);
end

if ~exist(zero_save_folder, 'dir')
    mkdir(zero_save_folder);
end

% Selected data file and saving folders
workingdir = [PALA_data_folder];
workingdir_1 = [PALA_data_folder 'DS_10_100HZ\'];
workingdir_2 = [PALA_data_folder 'DS_4_250HZ\'];
workingdir_3 = [PALA_data_folder 'DS_2_500HZ\'];
save_dir = [PALA_save_folder];
save_dir_1 = [save_dir '\US_10_100HZ_IMQ'];mkdir(save_dir_1)
save_dir_2 = [save_dir '\US_4_250HZ_IMQ'];mkdir(save_dir_2)
save_dir_3 = [save_dir '\US_2_500HZ_IMQ'];mkdir(save_dir_3)

zero_save_dir = [zero_save_folder];
zero_save_dir_1 = [zero_save_dir '\zero_US_10_100HZ'];mkdir(zero_save_dir_1)
zero_save_dir_2 = [zero_save_dir '\zero_US_4_250HZ'];mkdir(zero_save_dir_2)
zero_save_dir_3 = [zero_save_dir '\zero_US_2_500HZ'];mkdir(zero_save_dir_3)

filename = 'PALA_TEULM_';
cd(workingdir)

% D Load Original data
Ori_low_datas_100 = dir([workingdir_1 '*.mat']);
Ori_low_datas_250 = dir([workingdir_2 '*.mat']);
Ori_low_datas_500 = dir([workingdir_3 '*.mat']);
% Ori_low_datas_250 = Ori_low_datas_250(1:8);
% Ori_low_datas_100 = Ori_low_datas_100(1:8);
% Ori_low_datas_500 = Ori_low_datas_500(1:8);
%size(Ori_low_datas_100);

% Start algorithms
% B target data
Z = 1000;
cd(currentPath)

% epsilon shape parameter
epsilon = 1000; %IMQ:250,500->64,

% m condition value
m = 12;
% Start algorithms for each file
process_files(Ori_low_datas_100, 100, save_dir_1, epsilon, 'MQ', 11);
% process_files(Ori_low_datas_250, 250, save_dir_2, epsilon, 'IMQ', 5);
% process_files(Ori_low_datas_500, 500, save_dir_3, epsilon, 'IMQ', 3);
%data = load([Ori_low_datas_100(1).folder filesep Ori_low_datas_100(1).name]);
%data = data.data_100Hz;
%size(data)
%Phi = base_construct(data(1,:,1:2),'MQ', epsilon);
%zero_interpolate(Ori_low_datas_100, 100, zero_save_dir_1);
%zero_interpolate(Ori_low_datas_250, 250, zero_save_dir_2);
%zero_interpolate(Ori_low_datas_500, 500, zero_save_dir_3);
%data_100 = data_preprocess(Ori_low_datas_100, 100);
%Final_data = teulm_rbf_interpolation(data_100, m, epsilon, 'GA', 2, save_dir_1);
%zero_interpolate(Ori_low_datas_100, 100, save_dir_1);

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
    Interpolation_Set = zeros(78, 118, 800);
    % 100Hz interpolate 9 frame per frame
    if type == 100
        for i = 1: 80
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
        for i = 1: 200
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
        for i = 1: 400
            idx = (i - 1)* 2 + 1;
            Interpolation_Set(:,:, idx) = ori_data(:,:, i);
            % interpolate blank frame
            Interpolation_Set(:,:, idx + 1) = 0;
        end
    end
end

%% zero_interpolate
function zero_interpolate(file_list, type, save_dir)
    for i = 1:numel(file_list)
        % Load data
        data = load([file_list(i).folder filesep file_list(i).name]);
        
        switch type
            case 100
                single_data = data.data_100Hz;
            case 250
                single_data = data.data_250Hz;
            case 500
                single_data = data.data_500Hz;
        end
        
        % Perform interpolation
        IQ = init_interpolation_set(single_data, type);
        % Save interpolated data
        save([save_dir filesep 'data_' num2str(type) 'Hz_Up_' num2str(i) '.mat'], 'IQ');
    end
end

%% PROCESS
function process_files(file_list, type, save_dir, epsilon, RBF_type, target_frame)
    data_500 = load([file_list(1).folder filesep file_list(1).name]);
    switch type
        case 100
            data_500 = data_500.data_100Hz;
        case 250
            data_500 = data_500.data_250Hz;
        case 500
            data_500 = data_500.data_500Hz;
    end
    Phi = base_construct(data_500(1,:,1:2),RBF_type, epsilon);
    H  = h_construction(data_500(1,:,1:2), target_frame, RBF_type, epsilon);
    for i = 1:numel(file_list)
        % Load data
        data = load([file_list(i).folder filesep file_list(i).name]);
        
        switch type
            case 100
                single_data = data.data_100Hz;
            case 250
                single_data = data.data_250Hz;
            case 500
                single_data = data.data_500Hz;
        end
        
        % Perform interpolation
        IQ = teulm_rbf_interpolation(single_data, target_frame, Phi, H);
        % Save interpolated data
        save([save_dir filesep 'data_' num2str(type) 'Hz_Up_' num2str(i) '.mat'], 'IQ');
        disp(['===  Interpolation completed for file ' num2str(i) '!!! ===']);
    end
end

%% Base Construction
function Phi = base_construct(Data, RBF_Type, epsilon)
    % data size now 78 * 118 * 80
    % choose a random row 2D layer to construct base
    selected_row_data = squeeze(Data);     % 118 * 80 
    [num_cols, num_frames] = size(selected_row_data);   % C = 118, F = 80
    N = num_cols * num_frames;                          % N = 118 * 80
    Phi = zeros(N, N);
    size(Phi);
    normalized_cols = linspace(0, 1, num_cols);
    normalized_frames = linspace(0, 1, num_frames);
    
    % generate Phi
    for i = 1:num_frames
        for j = 1:num_cols
            for m = 1:num_frames
                for n = 1:num_cols
                    idx1 = (i-1) * num_cols + j;
                    idx2 = (m-1) * num_cols + n;
                    r = norm([normalized_frames(i), normalized_cols(j)] - [normalized_frames(m), normalized_cols(n)]);
                    Phi(idx1, idx2) = RBF(r, RBF_Type, epsilon);
                end
            end
        end
    end
    disp(['Size of Phi: ', num2str(size(Phi))])
    disp(['Rank of Phi: ', num2str(rank(Phi))]) %要不是无穷
    disp(['Cond of Phi: ', num2str(cond(Phi))]) %最低是1
    Phi = (Phi + 0.0001* eye(size(Phi))); %正则化
    disp(['new Rank of Phi: ', num2str(rank(Phi))])
    disp(['Cond of Phi: ', num2str(cond(Phi))])
end

%% F(IVTS) Construction
function IVTS = ivts_construct(Data, i)
    % extract
    layer_data = squeeze(Data(i, :, :));  
    % get 1D column vector
    IVTS = layer_data(:);  
    size(IVTS);
end

%% H Construction
function H = h_construction(ori_Data, new_frame_num, RBF_Type, epsilon)
    [~, col, old_frame_num] = size(ori_Data);           % 118 * 2             
    %[~, col, new_frame_num] = size(target_data);
    %new_frame_num = 11;
    H = zeros(col* new_frame_num, col* old_frame_num);  % M = col* new_frame_num, N = col* old_frame_num
    % Normalized scale
    normalized_cols = linspace(0, 1, col);
    normalized_frames = linspace(0, 1, old_frame_num);
    target_normalized_frames = linspace(0, 1, new_frame_num);
    % get H
    for i = 1: new_frame_num
        for j = 1: col % M
            for m = 1: old_frame_num
                for n = 1: col % N
                    idx1 = (i-1) * col + j;
                    idx2 = (m-1) * col + n;
                    r = norm([target_normalized_frames(i), normalized_cols(j)] - [normalized_frames(m), normalized_cols(n)]);
                    H(idx1, idx2) = RBF(r, RBF_Type, epsilon);
                end
            end
        end
    end
end

%% UPS 2D Interpolation
function target_data = teulm_rbf_interpolation(D, target_frame, Phi, H)
    % D is one file of dataset 78 * 118 * 80/200/400
    % epsilon m RBF_type you can choose randomly
    % type and batch_size are corresponding 
    % 100(Hz) 80(Frame); 250 200; 500 400
    [row, col, frame_num] = size(D);              % row = 78 col = 118 frame_num = 80/200/400...
    batch = (target_frame - 1);
    % batch 
    target_data = zeros(row, col, 800);
    for start_frame = 1: frame_num - 1
        end_frame = min(start_frame + 1, frame_num);
        for i = 1: row
            selected_data = D(:,:,start_frame:end_frame);
            IVTS_layer = ivts_construct(selected_data, i);
            Beta = Phi \ IVTS_layer;
            %size(Beta);
            f_pre = H * Beta;               % M * 1
            result = reshape(f_pre, col, []); % M = 118 * 800 col * new frame num
            target_data(i, :, (start_frame - 1)* batch + 1: start_frame * batch + 1) = result;    %%%%!!!!!!ups!!!!!!
            %disp(['===  Interpolation completed for layer ' num2str(i) '!!! ===']);
        end
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



