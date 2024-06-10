clear all;close('all')

%% Adjust path
% Define target data path
currentPath = pwd;
PALA_data_folder = [currentPath,'\PALA_data_InVivoRatBrain\'];
PALA_save_folder = [currentPath,'\PALA_TEULM\'];
% Selected data file and saving folders
workingdir = [PALA_data_folder];
save_dir = [PALA_save_folder];
filename = 'PALA_TEULM_';
cd(workingdir)

% Work path
mydatapath = [workingdir 'IQ\'];
IQfiles = dir([mydatapath '*.mat']);

% Save path
save_dir = [save_dir '\DS_DATA'];mkdir(save_dir)
workingdir_1 = [save_dir '\DS_10_100HZ'];mkdir(workingdir_1)
workingdir_2 = [save_dir '\DS_4_250HZ'];mkdir(workingdir_2)
workingdir_3 = [save_dir '\DS_2_500HZ'];mkdir(workingdir_3)

%% DS
%pars
Nbuffers = numel(IQfiles);          % number of bloc to process  = 240
h = waitbar(0, 'Download Sample Processing...');
for i = 1:Nbuffers
    waitbar(i/Nbuffers, h, ['Processing block ' num2str(i) ' of ' num2str(Nbuffers)]);
    % load data
    load([IQfiles(i).folder filesep IQfiles(i).name], 'IQ', 'PData', 'UF');
    
    % DS factor
    downsample_factors = [2, 4, 10];
    
    % [nb_pixel_z,nb_pixel_x, nb_frame_per_bloc] 
    % 78 * 118 * 800 every .mat
    [height, width, numFrames] = size(IQ);
    
    % ds
    data_500Hz = IQ(:, :, 1:2:end);  % DS=2
    data_250Hz = IQ(:, :, 1:4:end);  % DS=4
    data_100Hz = IQ(:, :, 1:10:end); % DS=10

    % update pars for UF
    UF_500Hz = UF;
    UF_250Hz = UF;
    UF_100Hz = UF;

    UF_500Hz.FrameRateUF = 500;
    UF_250Hz.FrameRateUF = 250;
    UF_100Hz.FrameRateUF = 100;

    UF_500Hz.NbFrames = 400; % 800/2
    UF_250Hz.NbFrames = 200; % 800/4
    UF_100Hz.NbFrames = 80;  % 800/10

    % save
    save([workingdir_3 filesep 'data_500Hz_' num2str(i) '.mat'], 'data_500Hz', "PData", "UF_500Hz");
    save([workingdir_2 filesep 'data_250Hz_' num2str(i) '.mat'], 'data_250Hz', "PData", "UF_250Hz");
    save([workingdir_1 filesep 'data_100Hz_' num2str(i) '.mat'], 'data_100Hz', "PData", "UF_100Hz");
end
close(h); % 关闭进度条
disp('=== Downsampling completed!!! ===');
% restore old path
cd(currentPath)
pwd
clear P UF IQ