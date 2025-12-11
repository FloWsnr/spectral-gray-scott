% Test script for save_gs_to_h5 function
% This will convert the existing simulation data to HDF5 format

% Add chebfun to path
addpath('chebfun');

% Convert existing simulation
pattern = 'gliders';
init = 'gaussians';
isim = 1;

fprintf('Testing HDF5 conversion...\n');
save_gs_to_h5(pattern, init, isim);

fprintf('\n');
fprintf('Verification:\n');

% Read back the HDF5 file to verify
subfolder = sprintf('snapshots/gs_%s_F=%.3d_k=%.3d_%s_%d', pattern, 1000*0.014, 1000*0.054, init, isim);
h5file = fullfile(subfolder, 'data.h5');

% Display HDF5 file info
fprintf('HDF5 file contents:\n');
info = h5info(h5file);
for i = 1:length(info.Datasets)
    fprintf('  Dataset: %s, Size: [', info.Datasets(i).Name);
    fprintf('%d ', info.Datasets(i).Dataspace.Size);
    fprintf(']\n');
end

fprintf('\nAttributes:\n');
for i = 1:length(info.Attributes)
    fprintf('  %s = %s\n', info.Attributes(i).Name, num2str(info.Attributes(i).Value));
end

fprintf('\nTest complete!\n');
