% Test script for gen_gs with direct HDF5 output
% This runs a very short simulation to test the HDF5 saving functionality

addpath('chebfun');

% Run a very short test simulation
pattern = 'gliders';
delta_u = 0.00002;
delta_v = 0.00001;
F = 0.014;
k = 0.054;
random_seed = 999;  % Use different seed for test
init_type = 'gaussians';
dt = 1;
snap_dt = 10;
tend = 20;  % Very short simulation for testing

fprintf('Running test simulation with HDF5 output...\n');
gen_gs(pattern, delta_u, delta_v, F, k, random_seed, init_type, dt, snap_dt, tend);

% Verify the output
subfolder = sprintf('snapshots/gs_%s_F=%.3d_k=%.3d_%s_%d', pattern, 1000*F, 1000*k, init_type, random_seed);
h5file = fullfile(subfolder, 'data.h5');

fprintf('\n========================================\n');
fprintf('Verification\n');
fprintf('========================================\n');

if exist(h5file, 'file')
    fprintf('HDF5 file created successfully!\n');
    fprintf('File: %s\n', h5file);

    % Get file info
    info = h5info(h5file);

    fprintf('\nDatasets:\n');
    for i = 1:length(info.Datasets)
        fprintf('  %s: [', info.Datasets(i).Name);
        fprintf('%d ', info.Datasets(i).Dataspace.Size);
        fprintf(']\n');
    end

    fprintf('\nAttributes:\n');
    for i = 1:length(info.Attributes)
        val = info.Attributes(i).Value;
        if isnumeric(val)
            fprintf('  %s = %s\n', info.Attributes(i).Name, num2str(val));
        else
            fprintf('  %s = %s\n', info.Attributes(i).Name, val);
        end
    end

    fprintf('\nTest PASSED!\n');
else
    fprintf('ERROR: HDF5 file was not created!\n');
    fprintf('Test FAILED!\n');
end

fprintf('========================================\n');

quit;
