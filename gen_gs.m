function gen_gs(pattern, delta_u, delta_v, F, k, random_seed, init_type)
% GEN_GS Generate a single Gray-Scott pattern simulation
%
% Syntax:
%   gen_gs(pattern, delta_u, delta_v, F, k, random_seed, init_type)
%
% Parameters:
%   pattern       - Pattern type: 'gliders', 'bubbles', 'maze', 'worms', 'spirals', 'spots'
%   delta_u       - Diffusion coefficient for u
%   delta_v       - Diffusion coefficient for v
%   F             - Feed rate parameter
%   k             - Kill rate parameter
%   random_seed   - Random seed for reproducibility (also used for output filename)
%   init_type     - Initialization type: 'gaussians' or 'fourier'

pattern = lower(pattern);
init_type = lower(init_type);

% Validate initialization type
if ~ismember(init_type, {'gaussians', 'fourier'})
    error('init_type must be either ''gaussians'' or ''fourier''');
end

% Set random seed
rng(random_seed);

% Simulation parameters
dom = [-1 1 -1 1];
n = 128;
dt = 1;
snap_dt = 10;
tend = 10000;
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
fprintf('Pattern:        %s\n', pattern);
fprintf('Initialization: %s\n', init_type);
fprintf('Random Seed:    %d\n', random_seed);
fprintf('Parameters:     F=%.4f, k=%.4f\n', F, k);
fprintf('Diffusion:      Du=%.5f, Dv=%.5f\n', delta_u, delta_v);
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

    % Create subfolder with parameter names
    subfolder = sprintf('snapshots/gs_%s_F=%.3d_k=%.3d_%s_%d', pattern, 1000*F, 1000*k, init_type, random_seed);
    if ~exist(subfolder, 'dir')
        mkdir(subfolder);
    end

    % Extract numerical data from chebfun objects for numpy/HDF5
    fprintf('Converting chebfun data to numerical arrays...\n');
    num_snapshots = length(uv);

    % Get values on a grid
    [xx, yy] = meshgrid(linspace(dom(1), dom(2), n), linspace(dom(3), dom(4), n));

    % Preallocate arrays
    u_snapshots = zeros(n, n, num_snapshots);
    v_snapshots = zeros(n, n, num_snapshots);

    for i = 1:num_snapshots
        u_snapshots(:,:,i) = uv{i}{1}(xx, yy);
        v_snapshots(:,:,i) = uv{i}{2}(xx, yy);
    end

    % Save as HDF5 (readable by numpy with h5py)
    h5file = fullfile(subfolder, 'data.h5');
    if exist(h5file, 'file')
        delete(h5file);
    end
    h5create(h5file, '/u', size(u_snapshots));
    h5create(h5file, '/v', size(v_snapshots));
    h5create(h5file, '/x', size(xx));
    h5create(h5file, '/y', size(yy));
    h5create(h5file, '/time', [num_snapshots, 1]);
    h5write(h5file, '/u', u_snapshots);
    h5write(h5file, '/v', v_snapshots);
    h5write(h5file, '/x', xx);
    h5write(h5file, '/y', yy);
    h5write(h5file, '/time', tspan(:));

    % Save metadata
    metadata.pattern = pattern;
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

    % Save metadata as JSON
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
    fprintf('  - data.h5 (HDF5/numpy format)\n');
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
