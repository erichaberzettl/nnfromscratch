from nn import *

rel_err = lambda a, n: abs(a - n) / max(abs(a) + abs(n), 1e-12)

def gradient_test():

    nn = NeuralNetwork([Layer(4, ReLu()), Layer(3,Identity())], CrossEntropy())
    X = np.random.normal(size=(4,3))
    Y = np.array([[0,0,1], [1,0,0], [1,0,0], [0,1,0]])
    nn.backward(nn.forward(X), Y)
    
    eps = 0.00001 # 1e-5
    errors = []
    for layer in nn.layers:
        # weights

        for i,j in np.ndindex(layer.w.shape):

            analytical_gradient = layer.dL_dw[i][j]
            original = layer.w[i][j]
            
            original_plus = original + eps
            layer.w[i][j] = original_plus
            loss_plus = nn.loss(nn.forward(X), Y)
            original_minus = original - eps
            layer.w[i][j] = original_minus
            loss_minus = nn.loss(nn.forward(X), Y)
            
            numerical_gradient = (loss_plus - loss_minus) / (2*eps)
            layer.w[i][j] = original
            
            if (err:=rel_err(analytical_gradient, numerical_gradient)) > np.float64(1e-6):
                errors.append({
                    "Parameter": f"w({i},{j})",
                    "Rel error": err
                })
            
        # biases
        for i,j in np.ndindex(layer.b.shape):

            analytical_gradient = layer.dL_db[i][j]
            original = layer.b[i][j]
            
            original_plus = original + eps
            layer.b[i][j] = original_plus
            loss_plus = nn.loss(nn.forward(X), Y)
            original_minus = original - eps
            layer.b[i][j] = original_minus
            loss_minus = nn.loss(nn.forward(X), Y)
            
            numerical_gradient = (loss_plus - loss_minus) / (2*eps)
            layer.b[i][j] = original
            
            if (err:= rel_err(analytical_gradient, numerical_gradient)) > np.float64(1e-6):
                errors.append({
                    "Parameter": f"b({i},{j})",
                    "Relerror": err
                    })
    print("All tests passed!") if not errors else print(errors)
                            
            
def cross_entropy_test():

    x = np.array([[-3, 1000, 3],[4,-1,2.2], [-2.23,3.3,0]], dtype="float64")
    y = np.array([[0,1,0,],[0,0,1],[1,0,0]])
    ce = CrossEntropy()
    eps = 0.00001
    errors = []
    for i,j in [(i,j) for i in range(3) for j in range(3)]:
        analytical_loss = ce.loss_gradient(x, y)[i][j]
        original = x[i][j] 
        
        x[i][j] = original + eps
        loss_plus = ce.loss_calculation(x, y)

        x[i][j] = original - eps
        loss_minus = ce.loss_calculation(x, y)
        
        numerical_loss = (loss_plus - loss_minus) / (2*eps)
        x[i,j] = original
        if (err:= rel_err(analytical_loss, numerical_loss)) > np.float64(1e-6):
                        errors.append({
                            "Parameter": f"b({i},{j})",
                            "Relerror": err
                            })
        print("All tests passed!") if not errors else print(errors)


