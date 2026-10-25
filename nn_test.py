from nn import *

def gradient_test():
    ...
    # test nn with test run
    nn = NeuralNetwork([Layer(4, ReLu()), Layer(3,Identity())], CrossEntropy())
    X = np.random.normal(size=(4,3))
    Y = np.array([[0,0,1], [1,0,0], [1,0,0], [0,1,0]])
    rel_err = lambda a, n: abs(a - n) / max(abs(a) + abs(n), 1e-12)

    nn.backward(Y, nn.forward(X))
    
    eps = 0.00001 # 1e-5
    errors = []
    for layer in nn.layers:
        # weights

        for i,j in np.ndindex(layer.w.shape):

            analytical_gradient = layer.dL_dw[i][j]
            original = layer.w[i][j]
            
            original_plus = original + eps
            layer.w[i][j] = original_plus
            loss_plus = nn.loss(Y, nn.forward(X))
            original_minus = original - eps
            layer.w[i][j] = original_minus
            loss_minus = nn.loss(Y, nn.forward(X))
            
            numerical_gradient = (loss_plus - loss_minus) / (2*eps)
            layer.w[i][j] = original
            
            if (rel_err(analytical_gradient, numerical_gradient)) > np.float64(1e-6):
                errors.append({
                    "Parameter": f"w(i,j)",
                    "Rel error": rel_err(analytical_gradient, numerical_gradient)
                })

            
        # biases
        for i,j in np.ndindex(layer.b.shape):

            analytical_gradient = layer.dL_db[i][j]
            original = layer.b[i][j]
            
            original_plus = original + eps
            layer.b[i][j] = original_plus
            loss_plus = nn.loss(Y, nn.forward(X))
            original_minus = original - eps
            layer.b[i][j] = original_minus
            loss_minus = nn.loss(Y, nn.forward(X))
            
            numerical_gradient = (loss_plus - loss_minus) / (2*eps)
            layer.b[i][j] = original
            
            if (rel_err(analytical_gradient, numerical_gradient)) > np.float64(1e-6):
                            errors.append({
                                "Parameter": f"b(i,j)",
                                "Relerror": rel_err(analytical_gradient, numerical_gradient)})
    print("All tests passed!") if not errors else print(errors)
                            
            
def cross_entropy_test():

    x = np.array([[-3, 1000, 3],[4,-1,2.2], [-2.23,3.3,0]], dtype="float64")
    y = np.array([[0,1,0,],[0,0,1],[1,0,0]])
    ce = CrossEntropy()
    eps = 0.00001

    for i,j in [(i,j) for i in range(3) for j in range(3)]:
        analytical_loss = ce.loss_gradient(y, x)[i][j]
        original = x[i][j] 
        
        x[i][j] = original + eps
        loss_plus = ce.loss_calculation(y, x)

        x[i][j] = original - eps
        loss_minus = ce.loss_calculation(y, x)

        
        numerical_loss = (loss_plus - loss_minus) / (2*eps)
        x[i,j] = original
        rel_err = lambda a, n: abs(a - n) / max(abs(a) + abs(n), 1e-12)
        print(f"({i},{j})")

        print(f"Analytical loss: {analytical_loss} vs. Numerical loss: {numerical_loss}")
        print(f"Relative error {(rel_err(analytical_loss, numerical_loss))}")
        print(f"Test passed? {(rel_err(analytical_loss, numerical_loss)) < np.float64(1e-6)}")
        print("-----------------------------------------------------------------", end="\n\n")


