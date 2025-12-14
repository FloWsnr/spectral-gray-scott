function gen_gs(delta_u, delta_v, F, k, random_seed, init_type, dt, snap_dt, tend, array_job_id, job_id)
% GEN_GS Generate a single Gray-Scott reaction-diffusion simulation
%
% Syntax:
%   gen_gs(delta_u, delta_v, F, k, random_seed, init_type, dt, snap_dt, tend, array_job_id, job_id)
%
% Parameters:
%   delta_u       - Diffusion coefficient for u
%   delta_v       - Diffusion coefficient for v
%   F             - Feed rate parameter
%   k             - Kill rate parameter
%   random_seed   - Random seed for reproducibility (also used for output filename)
%   init_type     - Initialization type: 'gaussians' or 'fourier'
%   dt            - Time step size (optional, default: 1)
%   snap_dt       - Snapshot interval (optional, default: 10)
%   tend          - Final time (optional, default: 10000)
%   array_job_id  - SLURM array job ID (optional, for directory naming)
%   job_id        - Job ID from parameter file (optional, for directory naming)

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
if nargin < 10 || isempty(array_job_id)
    array_job_id = '';
end
if nargin < 11 || isempty(job_id)
    job_id = '';
end

% Set random seed
rng(random_seed);

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

% Initialize based on type
if strcmp(init_type, 'gaussians')
    ngauss = [10 100];
    amp = [1 3];
    width = [150 300];
    normalize = @(u) (u-min2(u)) / max2(u-min2(u));

    uinit = random_gaussians(ngauss, amp, width, dom);
    uinit = @(x,y) 1-uinit(x,y);
    vinit = random_gaussians(ngauss, amp, width, dom);
    uinit = chebfun2(uinit, dom, 'trig');
    vinit = chebfun2(vinit, dom, 'trig');
    uinit = normalize(uinit);
    vinit = normalize(vinit);
    S.init = chebfun2v(uinit, vinit, dom);

    % Additional metadata for Gaussians
    metadata_extra.ngauss_range = ngauss;
    metadata_extra.amplitude_range = amp;
    metadata_extra.width_range = width;
else  % fourier
    nfourier = 32;
    [uinit, vinit] = init_fourier(F, k, nfourier, dom);
    S.init = chebfun2v(uinit, vinit, dom);

    % Additional metadata for Fourier
    metadata_extra.nfourier = nfourier;
end

fprintf('========================================\n');
fprintf('Gray-Scott Simulation Starting\n');
fprintf('========================================\n');
fprintf('Parameters:     F=%.4f, k=%.4f\n', F, k);
fprintf('Diffusion:      Du=%.5f, Dv=%.5f\n', delta_u, delta_v);
fprintf('Initialization: %s\n', init_type);
fprintf('Random Seed:    %d\n', random_seed);
fprintf('Domain:         [%.1f, %.1f] x [%.1f, %.1f]\n', dom(1), dom(2), dom(3), dom(4));
fprintf('Grid Size:      %d x %d\n', n, n);
fprintf('Time Step:      dt=%.2f\n', dt);
fprintf('Final Time:     t=%.0f\n', tend);
fprintf('Snapshots:      every %.0f time units (%d total)\n', snap_dt, length(tspan));
fprintf('Scheme:         %s\n', pref.scheme);
fprintf('========================================\n');
fprintf('Starting time integration...\n');
fprintf('This may take a while (simulating %.0f time units)...\n', tend);
fprintf('Start time: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('========================================\n\n');

try
    tic;
    uv = spin2(S, n, dt, pref);
    elapsed = toc;

    % Create subfolder based on array job ID and job ID if provided,
    % otherwise fall back to old naming scheme
    if ~isempty(array_job_id) && ~isempty(job_id)
        subfolder = sprintf('results/snapshots/%s/%s', array_job_id, job_id);
    else
        % Legacy naming for backward compatibility
        subfolder = sprintf('results/snapshots/gs_F=%.3d_k=%.3d_%s_%d', 1000*F, 1000*k, init_type, random_seed);
    end

    if ~exist(subfolder, 'dir')
        mkdir(subfolder);
    end

    % Convert chebfun data to arrays and save to HDF5
    fprintf('Converting chebfun data to arrays...\n');

    % Get dimensions
    [~, num_snapshots] = size(uv);
    fprintf('  Number of snapshots: %d\n', num_snapshots);

    % Set up grid for evaluation (Chebyshev points)
    x = linspace(dom(1), dom(2), n);
    y = linspace(dom(3), dom(4), n);
    [XX, YY] = meshgrid(x, y);

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

    % Save to HDF5 file
    h5file = fullfile(subfolder, 'data.h5');
    fprintf('Writing to HDF5 file: %s\n', h5file);

    % Delete existing HDF5 file if it exists
    if exist(h5file, 'file')
        delete(h5file);
    end

    % Write u and v datasets
    h5create(h5file, '/u', size(u_data));
    h5write(h5file, '/u', u_data);
    h5create(h5file, '/v', size(v_data));
    h5write(h5file, '/v', v_data);

    % Write spatial grids (Chebyshev points)
    h5create(h5file, '/x', size(x));
    h5write(h5file, '/x', x);
    h5create(h5file, '/y', size(y));
    h5write(h5file, '/y', y);

    % Write time array
    time_array = 0:snap_dt:tend;
    h5create(h5file, '/time', size(time_array));
    h5write(h5file, '/time', time_array);

    % Prepare metadata
    metadata.F = F;
    metadata.k = k;
    metadata.delta_u = delta_u;
    metadata.delta_v = delta_v;
    metadata.initialization = init_type;
    metadata.random_seed = random_seed;
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
    h5writeatt(h5file, '/', 'random_seed', random_seed);
    h5writeatt(h5file, '/', 'num_snapshots', num_snapshots);
    h5writeatt(h5file, '/', 'grid_size_x', n);
    h5writeatt(h5file, '/', 'grid_size_y', n);
    h5writeatt(h5file, '/', 'domain', dom);
    h5writeatt(h5file, '/', 'time_step', dt);
    h5writeatt(h5file, '/', 'snapshot_interval', snap_dt);
    h5writeatt(h5file, '/', 'final_time', tend);
    h5writeatt(h5file, '/', 'scheme', pref.scheme);
    h5writeatt(h5file, '/', 'dealias', pref.dealias);

    % Save metadata as JSON (for compatibility)
    jsonfile = fullfile(subfolder, 'metadata.json');
    fid = fopen(jsonfile, 'w');
    fprintf(fid, '%s', jsonencode(metadata));
    fclose(fid);

    fprintf('\n========================================\n');
    fprintf('Simulation Complete!\n');
    fprintf('========================================\n');
    fprintf('End time:       %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf('Elapsed time:   %.2f seconds (%.2f minutes)\n', elapsed, elapsed/60);
    fprintf('Saved to:       %s\n', subfolder);
    fprintf('Files created:\n');
    fprintf('  - data.h5 (HDF5 format with field values)\n');
    fprintf('    Datasets: /u [%dx%dx%d], /v [%dx%dx%d]\n', n, n, num_snapshots, n, n, num_snapshots);
    fprintf('  - metadata.json (JSON format)\n');
    fprintf('========================================\n');
catch ME
    fprintf('\n========================================\n');
    fprintf('ERROR: Simulation Failed!\n');
    fprintf('========================================\n');
    fprintf('Error message: %s\n', ME.message);
    fprintf('Error ID:      %s\n', ME.identifier);
    if ~isempty(ME.stack)
        fprintf('Error in:      %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('========================================\n');
    fprintf('This may indicate that the solution became unstable.\n');
    fprintf('Try adjusting parameters (dt, F, k) or initialization.\n');
    fprintf('========================================\n');
    warning('Solution blew up or error occurred: %s', ME.message);
end

end
