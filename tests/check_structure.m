% Check the structure of uv
addpath('chebfun');

file = 'snapshots/gs_gliders_F=014_k=054_gaussians_1/data.mat';
load(file, 'uv');

fprintf('Type of uv: %s\n', class(uv));
fprintf('Length of uv: %d\n', length(uv));
fprintf('Size of uv: [%d %d]\n', size(uv, 1), size(uv, 2));

% Try to access individual snapshots
fprintf('\nAccessing elements:\n');
fprintf('Type of uv{1,1}: %s\n', class(uv{1,1}));
fprintf('Type of uv{2,1}: %s\n', class(uv{2,1}));
fprintf('Type of uv{1,2}: %s\n', class(uv{1,2}));

fprintf('\nSo the structure is:\n');
fprintf('  uv{1,k} = u component at time snapshot k\n');
fprintf('  uv{2,k} = v component at time snapshot k\n');

% Try the syntax from the plot function
fprintf('\nTrying plot syntax uv{1}(:,:,ik):\n');
try
    % Try to access third dimension like in plot code
    test = uv{1}(:,:,1);
    fprintf('Type of uv{1}(:,:,1): %s\n', class(test));
    vals = real(test);
    fprintf('Real values size: [%d %d]\n', size(vals, 1), size(vals, 2));
catch ME
    fprintf('Error: %s\n', ME.message);
end

% Try using values() function directly
fprintf('\nTrying values() function:\n');
try
    u_component = uv{1};
    vals = values(u_component);
    fprintf('Type of vals: %s\n', class(vals));
    fprintf('Size of vals: [%d %d %d]\n', size(vals, 1), size(vals, 2), size(vals, 3));
catch ME
    fprintf('Error: %s\n', ME.message);
end

% Try to understand the internal structure
fprintf('\nChecking methods available:\n');
methods_list = methods(uv{1});
fprintf('Number of methods: %d\n', length(methods_list));

% Look for methods related to getting values
fprintf('\nMethods containing "val":\n');
for i = 1:length(methods_list)
    if contains(lower(methods_list{i}), 'val')
        fprintf('  %s\n', methods_list{i});
    end
end

% Try chebcoeffs2vals or get underlying values
fprintf('\nTrying to get coefficients/values:\n');
try
    % Try to access the underlying data directly
    u1 = uv{1};
    % Check if we can access .coeffs or .values fields
    if isprop(u1, 'coeffs')
        fprintf('Has coeffs property\n');
    end
    if isprop(u1, 'values')
        fprintf('Has values property\n');
    end

    % Try to evaluate on a grid
    dom = [-1 1 -1 1];
    n = 128;
    x = chebpts(n, dom(1:2));
    y = chebpts(n, dom(3:4));
    [XX, YY] = meshgrid(x, y);
    vals = u1(XX, YY);
    fprintf('Evaluated on grid: [%d %d]\n', size(vals, 1), size(vals, 2));
catch ME
    fprintf('Error: %s\n', ME.message);
end

quit;
