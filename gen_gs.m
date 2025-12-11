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

fprintf('Running %s initialization: seed=%d, F=%.4f, k=%.4f\n', ...
    init_type, random_seed, F, k);

try
    uv = spin2(S, n, dt, pref);
    file = sprintf('snapshots/gs_%s_F=%.3d_k=%.3d_%s_%d.mat', pattern, 1000*F, 1000*k, init_type, random_seed);
    save(file, 'uv');

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

    % Add initialization-specific metadata
    if exist('metadata_extra', 'var')
        fields = fieldnames(metadata_extra);
        for i = 1:length(fields)
            metadata.(fields{i}) = metadata_extra.(fields{i});
        end
    end

    metafile = sprintf('snapshots/gs_%s_F=%.3d_k=%.3d_%s_%d_metadata.mat', pattern, 1000*F, 1000*k, init_type, random_seed);
    save(metafile, 'metadata');

    fprintf('Successfully saved simulation with seed %d\n', random_seed);
catch ME
    warning('Solution blew up or error occurred: %s', ME.message);
end

end
