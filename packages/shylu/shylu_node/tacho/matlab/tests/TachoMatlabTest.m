clear
addpath '../bin'

%load sherman3.mtx; a = sherman3(2:end,:);
load qc_penrose_2_tc2.10.mtx; a = qc_penrose_2_tc2_10(2:end,:);
%load qc_penrose_10_tc0.30.mtx; a = qc_penrose_10_tc0_30(2:end,:);

A = sparse(a(:,1),a(:,2),a(:,3));
m = size(A,1);
n = size(A,2);
fprintf(1," A(%d x %d)\n",m,n);
b = randn(n,1);

tacho('option', 'verbose');
tacho('option', 'method', 'sk');
tacho('option', 'dofs-per-node', 2);
tacho('option', 'small-problem-thres', 200);

fprintf(1,' calling tachoMex(setup)\n');
  tacho('setup', A);
fprintf(1,' calling tachoMex(factorizee)\n');
  tacho('factor', A);
fprintf(1,' calling tachoMex(solve)\n');
  x = tacho('solve', b);
fprintf(1,' calling tachoMex(cleanup)\n');
  tacho('cleanup');
r = b - A*x;
nrmr = norm(r);
nrmb = norm(b);
fprintf(1,'residual = %e / %e = %e\n',nrmr,nrmb,nrmr/nrmb);

d = tacho('diag');
