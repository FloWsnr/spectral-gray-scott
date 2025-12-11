function save_gs_to_h5(pattern, init, isim)
% SAVE_GS_TO_H5 Convert Gray-Scott simulation data from .mat to HDF5 format
%
% Syntax:
%   save_gs_to_h5(pattern, init, isim)
%
% Parameters:
%   pattern - Pattern type: 'gliders', 'bubbles', 'maze', 'worms', 'spirals', 'spots'
%   init    - Initialization type: 'gaussians' or 'fourier'
%   isim    - Simulation number (random seed)
%
% This function loads the chebfun data from the .mat file and saves the
% real values to an HDF5 file with the same naming convention.

pattern = lower(pattern);

% Set F and k values based on pattern
switch pattern
    case 'gliders'
        F = 0.014;
        k = 0.054;
    case 'bubbles'
        F = 0.098;
        k = 0.057;
    case 'maze'
        F = 0.029;
        k = 0.057;
    case 'worms'
        F = 0.058;
        k = 0.065;
    case 'spirals'
        F = 0.018;
        k = 0.051;
    case 'spots'
        F = 0.03;
        k = 0.062;
    otherwise
        error('Unknown pattern.');
end

% Construct file paths
subfolder = sprintf('snapshots/gs_%s_F=%.3d_k=%.3d_%s_%d', pattern, 1000*F, 1000*k, init, isim);
matfile = fullfile(subfolder, 'data.mat');
h5file = fullfile(subfolder, 'data.h5');

fprintf('Loading data from: %s\n', matfile);

% Load the chebfun data
load(matfile, 'uv');

% Get dimensions
% uv is a chebmatrix of size [2, num_snapshots]
% uv{1, k} is u component at time snapshot k
% uv{2, k} is v component at time snapshot k
[~, num_snapshots] = size(uv);
fprintf('Number of snapshots: %d\n', num_snapshots);

% Set up grid for evaluation (same as in gen_gs.m)
dom = [-1 1 -1 1];
n = 128;
x = chebpts(n, dom(1:2));
y = chebpts(n, dom(3:4));
[XX, YY] = meshgrid(x, y);

fprintf('Grid size: %d x %d\n', n, n);

% Delete existing HDF5 file if it exists
if exist(h5file, 'file')
    delete(h5file);
    fprintf('Deleted existing HDF5 file\n');
end

% Create datasets for u and v
% HDF5 format: [nx, ny, num_snapshots]
fprintf('Converting chebfun data to arrays...\n');

% Preallocate arrays
u_data = zeros(n, n, num_snapshots);
v_data = zeros(n, n, num_snapshots);

% Extract values by evaluating chebfun2 objects on the grid
for ik = 1:num_snapshots
    u_data(:,:,ik) = real(uv{1, ik}(XX, YY));
    v_data(:,:,ik) = real(uv{2, ik}(XX, YY));

    if mod(ik, 10) == 0 || ik == num_snapshots
        fprintf('  Processed snapshot %d/%d\n', ik, num_snapshots);
    end
end

fprintf('  Conversion complete\n');

% Write to HDF5 file
fprintf('Writing to HDF5 file: %s\n', h5file);

% Write u data
h5create(h5file, '/u', size(u_data));
h5write(h5file, '/u', u_data);

% Write v data
h5create(h5file, '/v', size(v_data));
h5write(h5file, '/v', v_data);

% Add metadata as attributes
h5writeatt(h5file, '/', 'pattern', pattern);
h5writeatt(h5file, '/', 'F', F);
h5writeatt(h5file, '/', 'k', k);
h5writeatt(h5file, '/', 'initialization', init);
h5writeatt(h5file, '/', 'random_seed', isim);
h5writeatt(h5file, '/', 'num_snapshots', num_snapshots);
h5writeatt(h5file, '/', 'grid_size_x', n);
h5writeatt(h5file, '/', 'grid_size_y', n);
h5writeatt(h5file, '/', 'domain', dom);

fprintf('========================================\n');
fprintf('Conversion Complete!\n');
fprintf('========================================\n');
fprintf('HDF5 file: %s\n', h5file);
fprintf('Datasets:\n');
fprintf('  /u : %d x %d x %d (u field values)\n', n, n, num_snapshots);
fprintf('  /v : %d x %d x %d (v field values)\n', n, n, num_snapshots);
fprintf('Metadata stored as HDF5 attributes\n');
fprintf('========================================\n');

end
