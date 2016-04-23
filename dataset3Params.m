function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.3;
value = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

for i = 1:size(value,1)
    C = value(i);
    for j = 1 : size(value,1)
        
        sigma = value(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        eval_values(i,j)  = mean(double(predictions ~= yval));
    end
end
[M,I] = min(eval_values(:));
[I_row, I_col] = ind2sub(size(eval_values),I);  

C = value(I_row);
sigma = value(I_col);


end
