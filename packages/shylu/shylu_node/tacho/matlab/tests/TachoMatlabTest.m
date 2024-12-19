clear
addpath '../bin'

load sherman3.mtx
A = sparse(sherman3(2:end,1),sherman3(2:end,2),sherman3(2:end,3));
m = size(A,1);
n = size(A,2);
fprintf(1," A(%d x %d)\n",m,n);
b = randn(n,1);

tacho('option', 'verbose');
tacho('option', 'method', 'chol');

fprintf(1,' calling tacho(setup)\n');
  tacho('setup', A);
fprintf(1,' calling tacho(solve)\n');
  x = tacho('solve', b);
fprintf(1,' calling tacho(cleanup)\n');
  tacho('cleanup');
r = b - A*x;
nrmr = norm(r);
nrmb = norm(b);
fprintf(1,'residual = %e / %e = %e\n',nrmr,nrmb,nrmr/nrmb);
