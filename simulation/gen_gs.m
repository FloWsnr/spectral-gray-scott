function gen_gs(delta_u, delta_v, F, k, random_seeds, init_type, dt, snap_dt, tend)
% GEN_GS Generate Gray-Scott reaction-diffusion simulations for multiple random seeds
%
% Syntax:
%   gen_gs(delta_u, delta_v, F, k, random_seeds, init_type, dt, snap_dt, tend)
%
% Parameters:
%   delta_u       - Diffusion coefficient for u
%   delta_v       - Diffusion coefficient for v
%   F             - Feed rate parameter
%   k             - Kill rate parameter
%   random_seeds  - Array of random seeds (e.g., [1,2,3,...,100]) or scalar
%   init_type     - Initialization type: 'gaussians' or 'fourier'
%   dt            - Time step size (optional, default: 1)
%   snap_dt       - Snapshot interval (optional, default: 10)
%   tend          - Final time (optional, default: 10000)

init_type = lower(init_type);

% Validate initialization type
if ~ismember(init_type, {'gaussians', 'fourier'})
    error('init_type must be either ''gaussians'' or ''fourier''');
end

% Set default values for optional time parameters
if nargin < 7 || isempty(dt)
    dt = 1;
end
if nargin < 8 || isempty(snap_dt)
    snap_dt = 10;
end
if nargin < 9 || isempty(tend)
    tend = 10000;
end

% Validate and prepare seed array
if isempty(random_seeds)
    error('random_seeds cannot be empty');
end

% Convert scalar to array for consistency
if isscalar(random_seeds)
    random_seeds = [random_seeds];
end

n_seeds = length(random_seeds);

% Simulation parameters
dom = [-1 1 -1 1];
n = 128;
tspan = 0:snap_dt:tend;

pref = spinpref2();
pref.plot = 'off';
pref.scheme = 'etdrk4';
pref.dealias = 'on';

% Create operator
S = spinop2(dom, tspan);
S.lin    = @(u,v) [ delta_u*lap(u)   ; delta_v*lap(v)  ];
S.nonlin = @(u,v) [ -u.*v.^2+F*(1-u) ; u.*v.^2-(F+k)*v ];

% Store initialization parameters for later use in loop
if strcmp(init_type, 'gaussians')
    ngauss = [10 100];
    amp = [1 3];
    width = [150 300];
    normalize = @(u) (u-min2(u)) / max2(u-min2(u));

    % Additional metadata for Gaussians
    metadata_extra.ngauss_range = ngauss;
    metadata_extra.amplitude_range = amp;
    metadata_extra.width_range = width;
else  % fourier
    nfourier = 32;

    % Additional metadata for Fourier
    metadata_extra.nfourier = nfourier;
end

fprintf('========================================\n');
fprintf('Gray-Scott Simulation Starting\n');
fprintf('========================================\n');
fprintf('Parameters:     F=%.4f, k=%.4f\n', F, k);
fprintf('Diffusion:      Du=%.5f, Dv=%.5f\n', delta_u, delta_v);
fprintf('Initialization: %s\n', init_type);
fprintf('Random Seeds:   %d seeds (min=%d, max=%d)\n', n_seeds, min(random_seeds), max(random_seeds));
fprintf('Domain:         [%.1f, %.1f] x [%.1f, %.1f]\n', dom(1), dom(2), dom(3), dom(4));
fprintf('Grid Size:      %d x %d\n', n, n);
fprintf('Time Step:      dt=%.2f\n', dt);
fprintf('Final Time:     t=%.0f\n', tend);
fprintf('Snapshots:      every %.0f time units (%d total)\n', snap_dt, length(tspan));
fprintf('Scheme:         %s\n', pref.scheme);
fprintf('========================================\n');
fprintf('Starting time integration for %d trajectories...\n', n_seeds);
fprintf('This may take a while (simulating %.0f time units × %d seeds)...\n', tend, n_seeds);
fprintf('Start time: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('========================================\n\n');

tic;

% Pre-allocate arrays after first simulation
uv_all = [];
first_run = true;

% Track successful simulations
successful_indices = [];
failed_seeds = [];

% Loop over random seeds
for seed_idx = 1:n_seeds
    current_seed = random_seeds(seed_idx);
    fprintf('\n----------------------------------------\n');
    fprintf('Trajectory %d/%d (seed=%d)\n', seed_idx, n_seeds, current_seed);
    fprintf('----------------------------------------\n');

    % Set random seed for this trajectory
    rng(current_seed);

    % Re-initialize S.init for this seed
    if strcmp(init_type, 'gaussians')
        uinit = random_gaussians(ngauss, amp, width, dom);
        uinit = @(x,y) 1-uinit(x,y);
        vinit = random_gaussians(ngauss, amp, width, dom);
        uinit = chebfun2(uinit, dom, 'trig');
        vinit = chebfun2(vinit, dom, 'trig');
        uinit = normalize(uinit);
        vinit = normalize(vinit);
        S.init = chebfun2v(uinit, vinit, dom);
    else
        [uinit, vinit] = init_fourier(F, k, nfourier, dom);
        S.init = chebfun2v(uinit, vinit, dom);
    end

    % Run simulation with error handling
    try
        fprintf('Running ETDRK4 integration...\n');
        sim_start = tic;
        uv = spin2(S, n, dt, pref);
        sim_elapsed = toc(sim_start);
        fprintf('Simulation %d/%d complete in %.2f seconds (%.2f minutes)\n', seed_idx, n_seeds, sim_elapsed, sim_elapsed/60);

        % Extract data
        [~, num_snapshots] = size(uv);

        % Allocate storage on first iteration
        if first_run
            fprintf('Allocating storage for [%d seeds × %d snapshots × %d × %d × 2 channels]...\n', ...
                    n_seeds, num_snapshots, n, n);
            uv_all = zeros(n_seeds, n, n, num_snapshots, 2, 'single');
            first_run = false;

            % Set up grid for evaluation
            x = linspace(dom(1), dom(2), n);
            y = linspace(dom(3), dom(4), n);
            [XX, YY] = meshgrid(x, y);
        end

        % Extract chebfun2 values to array
        fprintf('Extracting chebfun2 values...\n');
        for ik = 1:num_snapshots
            uv_all(seed_idx, :, :, ik, 1) = single(real(uv{1, ik}(XX, YY)));
            uv_all(seed_idx, :, :, ik, 2) = single(real(uv{2, ik}(XX, YY)));

            if mod(ik, 200) == 0 || ik == num_snapshots
                fprintf('  Seed %d: Processed snapshot %d/%d\n', current_seed, ik, num_snapshots);
            end
        end

        % Mark this simulation as successful
        successful_indices(end+1) = seed_idx;

    catch ME
        fprintf('\n========================================\n');
        fprintf('ERROR: Simulation Failed for seed %d!\n', current_seed);
        fprintf('========================================\n');
        fprintf('Error message: %s\n', ME.message);
        fprintf('Error ID:      %s\n', ME.identifier);
        if ~isempty(ME.stack)
            fprintf('Error in:      %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('========================================\n');
        fprintf('This may indicate that the solution became unstable.\n');
        fprintf('Continuing with remaining seeds...\n');
        fprintf('========================================\n');
        warning('Seed %d failed: %s', current_seed, ME.message);

        % Track failed seed
        failed_seeds(end+1) = current_seed;
    end
end

total_elapsed = toc;

% Filter out failed trajectories
n_successful = length(successful_indices);
n_failed = length(failed_seeds);

fprintf('\n========================================\n');
fprintf('Simulation Summary\n');
fprintf('========================================\n');
fprintf('Total seeds attempted: %d\n', n_seeds);
fprintf('Successful:            %d\n', n_successful);
fprintf('Failed:                %d\n', n_failed);
if n_failed > 0
    fprintf('Failed seed IDs:       %s\n', mat2str(failed_seeds));
end
fprintf('Total elapsed time:    %.2f seconds (%.2f minutes)\n', total_elapsed, total_elapsed/60);
if n_successful > 0
    fprintf('Average time per trajectory: %.2f seconds\n', total_elapsed/n_successful);
end
fprintf('========================================\n');

% Exit if no successful simulations
if n_successful == 0
    error('All simulations failed! No data to save.');
end

% Keep only successful trajectories
if n_failed > 0
    fprintf('\nRemoving %d failed trajectories from data...\n', n_failed);
    uv_all = uv_all(successful_indices, :, :, :, :);
    random_seeds = random_seeds(successful_indices);
    n_seeds = n_successful;  % Update count
    fprintf('Final data shape: [%d successful trajectories × %d × %d × %d × 2]\n', ...
            n_seeds, n, n, num_snapshots);
end

% Create subfolder based on parameters only (excluding random seeds)
subfolder = sprintf('results/snapshots/F%.3f_k%.3f_du%.1e_dv%.1e_%s', ...
                    F, k, delta_u, delta_v, init_type);

if ~exist(subfolder, 'dir')
    mkdir(subfolder);
end

% Save to HDF5 file
h5file = fullfile(subfolder, 'data.h5');
fprintf('Writing to HDF5 file: %s\n', h5file);

% Delete existing HDF5 file if it exists
if exist(h5file, 'file')
    delete(h5file);
end

% Permute dimensions: [n_seeds, n, n, num_snapshots, 2] → [2, n, n, num_snapshots, n_seeds]
% MATLAB writes in column-major, so Python will read the reversed shape:
% Python reads: [n_seeds, num_snapshots, n, n, 2] = [n_trajectories, n_time, x, y, channels]
fprintf('Permuting dimensions for HDF5 output...\n');
uv_out = permute(uv_all, [5, 2, 3, 4, 1]);

fprintf('  MATLAB shape: [%d, %d, %d, %d, %d] (will be reversed in Python)\n', size(uv_out));
fprintf('  Python will read: [n_trajectories=%d, n_time=%d, x=%d, y=%d, channels=%d]\n', n_seeds, num_snapshots, n, n, 2);

% Convert spatial grids to single precision
x = single(x);
y = single(y);

% Write with compression (deflate level 5)
% ChunkSize matches MATLAB shape [2, n, n, num_snapshots, 1] to chunk by trajectory
fprintf('Writing datasets with compression...\n');
h5create(h5file, '/uv', size(uv_out), 'Datatype', 'single', ...
         'ChunkSize', [2, n, n, num_snapshots, 1], 'Deflate', 5);
h5write(h5file, '/uv', uv_out);

% Write spatial grids
h5create(h5file, '/x', size(x), 'Datatype', 'single');
h5write(h5file, '/x', x);
h5create(h5file, '/y', size(y), 'Datatype', 'single');
h5write(h5file, '/y', y);

% Write time array
time_array = single(0:snap_dt:tend);
h5create(h5file, '/time', size(time_array), 'Datatype', 'single');
h5write(h5file, '/time', time_array);

% Write random_seeds as dataset (cannot use attributes for arrays in old MATLAB)
h5create(h5file, '/random_seeds', size(random_seeds), 'Datatype', 'int32');
h5write(h5file, '/random_seeds', int32(random_seeds));

fprintf('HDF5 write complete.\n');

% Prepare metadata
metadata.F = F;
metadata.k = k;
metadata.delta_u = delta_u;
metadata.delta_v = delta_v;
metadata.initialization = init_type;
metadata.random_seeds = random_seeds;     % Changed: now array
metadata.n_trajectories = n_seeds;        % NEW field
metadata.domain = dom;
metadata.grid_size = n;
metadata.time_step = dt;
metadata.snapshot_interval = snap_dt;
metadata.final_time = tend;
metadata.scheme = pref.scheme;
metadata.dealias = pref.dealias;
metadata.num_snapshots = num_snapshots;

% Add initialization-specific metadata
if exist('metadata_extra', 'var')
    fields = fieldnames(metadata_extra);
    for i = 1:length(fields)
        metadata.(fields{i}) = metadata_extra.(fields{i});
    end
end

% Write metadata as HDF5 attributes
h5writeatt(h5file, '/', 'F', F);
h5writeatt(h5file, '/', 'k', k);
h5writeatt(h5file, '/', 'delta_u', delta_u);
h5writeatt(h5file, '/', 'delta_v', delta_v);
h5writeatt(h5file, '/', 'initialization', init_type);
h5writeatt(h5file, '/', 'n_trajectories', n_seeds);  % NEW
h5writeatt(h5file, '/', 'num_snapshots', num_snapshots);
h5writeatt(h5file, '/', 'grid_size_x', n);
h5writeatt(h5file, '/', 'grid_size_y', n);
h5writeatt(h5file, '/', 'domain', dom);
h5writeatt(h5file, '/', 'time_step', dt);
h5writeatt(h5file, '/', 'snapshot_interval', snap_dt);
h5writeatt(h5file, '/', 'final_time', tend);
h5writeatt(h5file, '/', 'scheme', pref.scheme);
h5writeatt(h5file, '/', 'dealias', pref.dealias);
% Note: random_seeds array written as dataset (see above)

% Save metadata as JSON (for compatibility)
jsonfile = fullfile(subfolder, 'metadata.json');
fid = fopen(jsonfile, 'w');
fprintf(fid, '%s', jsonencode(metadata));
fclose(fid);

fprintf('\n========================================\n');
fprintf('All Simulations Complete!\n');
fprintf('========================================\n');
fprintf('End time:       %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('Elapsed time:   %.2f seconds (%.2f minutes)\n', total_elapsed, total_elapsed/60);
fprintf('Trajectories:   %d\n', n_seeds);
fprintf('Saved to:       %s\n', subfolder);
fprintf('Files created:\n');
fprintf('  - data.h5 (HDF5 format with field values)\n');
fprintf('    Dataset: /uv [%d×%d×%d×%d×%d]\n', n_seeds, num_snapshots, n, n, 2);
fprintf('    Dimensions: [n_trajectories, n_time, x, y, channels]\n');
fprintf('    Channels: 0=u, 1=v\n');
fprintf('    Compression: Deflate level 5\n');
fprintf('  - metadata.json (JSON format)\n');
fprintf('========================================\n');

end
