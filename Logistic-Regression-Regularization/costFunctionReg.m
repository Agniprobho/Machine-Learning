function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


for i=1:m
    p = theta'*X(i,:)';
    J = J + (1/m)*(-y(i)*log(sigmoid(p))-(1-y(i))*log(1-sigmoid(p)));
end

[a,b]=size(theta);

for j=1:a
    for i=1:m
        p = theta'*X(i,:)';
        grad(j,1) = grad(j,1) + (1/m)*(sigmoid(p)-y(i))*X(i,j);
    end
end

for i=2:a
    J = J + (lambda/(2*m))*theta(i,1)^2;
    grad(i,1) = grad(i,1) + (lambda/m)*theta(i);
end



% =============================================================

end
