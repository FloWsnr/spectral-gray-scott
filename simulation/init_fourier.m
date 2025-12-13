function [u, v] = init_fourier(F, k, n, dom)

meanu = 1;
meanv = 0;
du = 1;
dv = 1;

% Perturb about the third steady state
if ( k < (sqrt(F)-2*F)/2 )
    A = sqrt(F) / (F+k);
    meanu = real((A-sqrt(A^2-4))/(2*A));
    meanv = real(sqrt(F)*(A+sqrt(A^2-4))/2);
    du = 0.9*min(meanu, 1-meanu);
    dv = 0.9*min(meanv, 1-meanv);
end

u = meanu + 0.5*du*chebfun2(rand(n)-0.5, dom, 'trig');
v = meanv + 0.5*dv*chebfun2(rand(n)-0.5, dom, 'trig');

end
