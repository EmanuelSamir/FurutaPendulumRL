function y_hat = Feedforward(inputs, W1, W2, W3)
    z_2 = inputs * W1;
    a_2 = ActFunc(z_2, ActFuncType);

    z_3 = a_2 * W2;
    a_3 = ActFunc(z_3, ActFuncType);

    z_4 = a_3 * W3;
    y_hat = ActFunc(z_4, 2);
    
    
function a = ActFunc(z, ActFuncType)
    a = 0;
    if ActFuncType == 1
        a = 1./(1+exp(-z));
    end
    if ActFuncType == 2
       a = z; 
    end
end

end
