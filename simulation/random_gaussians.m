function u = random_gaussians(mrange, arange, srange, dom)

if ( length(mrange) == 1 )
    mrange = [mrange mrange];
end
if ( length(arange) == 1 )
    arange = [arange arange];
end
if ( length(srange) == 1 )
    srange = [srange srange];
end

m = randi(mrange);
a = arange(1) + diff(arange)*rand(m,1);
s = srange(1) + diff(srange)*rand(m,1);

dx = diff(dom(1:2));
dy = diff(dom(3:4));
cx = dom(1) + dx*rand(m,1);
cy = dom(3) + dy*rand(m,1);

% Periodic tiling
a = repmat(a, 9, 1);
s = repmat(s, 9, 1);
cx = cx + [-dx -dx -dx  0   0   0   dx  dx  dx]; cx = cx(:);
cy = cy + [ dy  0  -dy  dy  0  -dy  dy  0  -dy]; cy = cy(:);

a  = reshape(a,  [1 1 9*m]);
s  = reshape(s,  [1 1 9*m]);
cx = reshape(cx, [1 1 9*m]);
cy = reshape(cy, [1 1 9*m]);

u = @(x,y) sum(a.*exp(-s.*((x-cx).^2 + (y-cy).^2)), 3);

end
