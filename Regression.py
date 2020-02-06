import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from numpy.linalg import inv

## Regression Analysis
# Data Generation

def f_true(x):
    y = 6.0 * (np.sin(x + 2) + np.sin(2 * x + 4))
    return y

n = 750
X = np.random.uniform(-7.5,7.5,n)
e = np.random.normal(0.0,5.0,n)
y = f_true(X) + e

plt.figure()
plt.scatter(X,y,12,marker='o')

X_true = np.arange(-7.5,7.5,0.05)
y_true = f_true(X_true)

plt.plot(X_true,y_true,marker='None',color='r')
plt.show()

test_frac = 0.3
val_frac = 0.1
X_trn, X_test, y_trn, y_test = tts(X,y,test_size=test_frac,random_state=42)
X_trn, X_val, y_trn, y_val = tts(X_trn, y_trn, test_size=val_frac,random_state=42)

plt.figure()
plt.scatter(X_trn,y_trn,12,marker='o',color='orange')
plt.scatter(X_val,y_val,12,marker='o',color='green')
plt.scatter(X_test,y_test,12,marker='o',color='blue')
plt.show()

##   Question 1
# Regression with polynomial function

#1 (a)

def polynomial_transform(X, d) :
    phi = np.vander(X,d,increasing=True)
    return phi


#1 (b)

def train_model(Phi, y) :
    PhiTrans = np.transpose(Phi)
    w = np.matmul(np.matmul(inv(np.matmul(PhiTrans,Phi)),PhiTrans),y)
    return w


#1 (c)

def evaluate_model(Phi, y, w) :
    n = len(y)
    mse = np.sum(np.square((y - (np.dot(Phi,w.reshape(-1,1))))))
    mse = mse/n
    return mse


w = {}
validationErr = {}
testErr = {}

for d in range(3,25,3) :
    Phi_trn = polynomial_transform(X_trn,d)
    w[d] = train_model(Phi_trn,y_trn)

    Phi_val = polynomial_transform(X_val,d)
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])

    Phi_test = polynomial_transform(X_test,d)
    testErr[d] = evaluate_model(Phi_test, y_test, w[d])

plt.figure()
plt.plot(list(validationErr.keys()),list(validationErr.values()),marker='o',linewidth=3,markersize=12)
plt.plot(list(testErr.keys()),list(testErr.values()),marker='s',linewidth=3,markersize=12)

plt.xlabel('Polynomial degree',fontsize=16)
plt.ylabel('Validation/Test Error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error','Test Error'], fontsize=16)
plt.axis('auto')
plt.show()

plt.figure()
plt.plot(X_true,y_true,marker='None',linewidth=5,color='k')

for d in range(9,25,3) :
    X_d = polynomial_transform(X_true,d)
    y_d = X_d.dot(w[d])
    plt.plot(X_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9,25,3)))
plt.axis([-8,8,-15,15])
plt.show()


#############################################################################


## Question 2
# Regression with radial basis function

#2 (a)

def radial_basis_transform(X, B, gamma=0.1):
    Phi = np.empty((len(X),len(B)))
    for i in range(0,len(X),1) :
        for j in range(0,len(B),1) :
            Phi[i,j] = np.exp(-gamma * np.square(X[i]-B[j]))
    return Phi

#2 (b)

def train_ridge_model(Phi, y, lam) :
    PhiTran = np.transpose(Phi)
    idenMat = np.identity(len(y))
    w = np.matmul(inv(np.matmul(PhiTran,Phi) + (idenMat * lam)),np.matmul(PhiTran,y))
    return w

#2 (c)

w_radial = {}
validationErr_radial = {}
testErr_radial = {}

for lam in (10**i for i in range(-3,3,1)) :
    Phi_trn_radial = radial_basis_transform(X_trn,X_trn)
    w_radial[lam] = train_ridge_model(Phi_trn_radial,y_trn,lam)

    Phi_val_radial = radial_basis_transform(X_val,X_trn)
    validationErr_radial[lam] = evaluate_model(Phi_val_radial,y_val,w_radial[lam])

    Phi_test_radial = radial_basis_transform(X_test,X_trn)
    testErr_radial[lam] = evaluate_model(Phi_test_radial,y_val,w_radial[lam])

plt.figure()
plt.plot(list(validationErr_radial.keys()),list(validationErr_radial.values()),marker='o',linewidth=3,markersize=12)
plt.plot(list(testErr_radial.keys()),list(testErr_radial.values()),marker='s',linewidth=3,markersize=12)
plt.xlabel('Lambda',fontsize=16)
plt.ylabel('Validation/Test Error', fontsize=16)
plt.xticks(list(validationErr_radial.keys()), fontsize=12)
plt.axis('auto')
plt.show()

print("The ideal values lambda can take are 100,10,1.")

#2 (d)

plt.figure()
plt.plot(X_true,y_true,marker='None',linewidth=5,color='k')

for lam in (10**i for i in range(-3,3,1)) :
    X_lam = radial_basis_transform(X_true,X_trn)
    y_lam = X_lam @ w_radial[lam]
    plt.plot(X_true, y_lam, marker='None', linewidth=2)

plt.legend(['true'] + list(validationErr_radial.keys()))
plt.show()

print("Linearity of the model increases as lambda increses")
