function a = ActFunc(z, ActFuncType)
    a = 0;
    if ActFuncType == 1
        a = 1./(1+exp(-z));
    end
    if ActFuncType == 2
       a = z; 
    end
end