function varargout = tacho(varargin)
%
% Call Syntax:
% (1) h = tacho('setup', A [, 'parameter', value,...]) - setup
%  Input:
%  A               - Matrix to be solved with (must be sparse)
%  parameter       - Teuchos parameter key (string)
%  value           - Value for Teuchos parameter
%  Output:
%  h               - The Mex handle for the system
%
% (2) x = tacho(h, A, b[, 'parameter', value,...])
%  Input:
%  h               - The TachoMex handle for the system to solve (int)
%  A               - Matrix to be solved with
%  b               - The RHS to solve with (vector or multivector)
%  parameter       - Teuchos parameter key (string)
%  value           - Value for Teuchos parameter
%  Output:
%  x               - Solution (vector or multivector)
%
% (3) x = tacho(h, b[, 'parameter', value,...])
%  Input:
%  h               - The TachoMex handle for the system to solve (int)
%  b               - The RHS to solve with (vector or multivector)
%  parameter       - Teuchos parameter key (string)
%  value           - Value for Teuchos parameter
%  Output:
%  x               - Solution (vector or multivector)
%
%  In this case the original 'A' matrix with which this was
%  constructed is now the coefficient matrix.
%
% (4) x = tacho('solve', h, r)
%  Input:
%  h               - The TachoMex handle for the system to solve (int)
%  r               - The residual/RHS vector to solve with (vector or multivector)
%  Output:
%  x               - M*r, where M is the preconditioner operator
%
% (5) muelu('cleanup'[, h]) - frees allocated memory
%  Input:
%  h               - The TachoMex handle for the system to clean up.
%                   Calling 'cleanup' with no handle cleans up all
%                   the systems.
%

if(strcmp(varargin{1},'setup')),
    % Setup mode = 0
    tachoMex(0, varargin{2:nargin});
elseif(strcmp(varargin{1}, 'factor')),
    % Factor mode = 1
    tachoMex(1, varargin{2:nargin});
elseif(strcmp(varargin{1}, 'solve')),
    % Solve mode = 2
    varargout{1} = tachoMex(2, varargin{2:nargin});
elseif(strcmp(varargin{1}, 'diag')),
    % Diag mode = 5
    varargout{1} = tachoMex(5, varargin{2:nargin});
elseif(strcmp(varargin{1}, 'pfaffian')),
    % Pfaffian mode = 7
    varargout{1} = tachoMex(7, varargin{2:nargin});
elseif(strcmp(varargin{1}, 'cleanup')),
    % Cleanup mode = 3
    tachoMex(3);
elseif(strcmp(varargin{1}, 'option')),
    % Option mode = 4
    tachoMex(4, varargin{2:nargin});
else
    fprintf('\nUsage:\n');
    fprintf('h = tachoMex(''setup'', A) to setup a problem\n');
    fprintf('h = tachoMex(''factor'', A) to factor the matrix\n');
    fprintf('x = tachoMex(''solve'', b) to solve the problem #h with the RHS vector b\n');
    fprintf('tachoMex(''cleanup'') to free memory associated with all problems, or a specific one.\n');
end
end
