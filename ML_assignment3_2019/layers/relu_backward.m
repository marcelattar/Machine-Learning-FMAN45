function dldx = relu_backward(x, dldy)
    sympref('HeavisideAtOrigin', 0);
    dldx = dldy.*heaviside(x);
end
