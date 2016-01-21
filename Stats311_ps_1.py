import numpy as np
import scipy
import matplotlib.pyplot as plt  
import matplotlib.patches as mpatches


'''
    Returns a random unit vector, dim is the dimension of the vector. high sigma, better the randomness 
'''
def random_unit_norm(dim,sigma = 100):
    mean = np.zeros([dim])
    cor  = sigma*np.identity(dim)
    gaussian    = np.random.multivariate_normal(mean, cor)
    unit_vector = gaussian/np.linalg.norm(gaussian)
    return unit_vector

'''
    Return the eigen vector corresponding to the maximum eigen value
'''
def max_eigen_value(Array):
#     Array = np.identity(Array.shape[0])
#     Array[2][2] = 2
    eig_val, eig_vec =   np.linalg.eigh(Array)
    max_index = np.any( np.argmax(eig_val) )
    return eig_vec[max_index]

'''
    Choice of w_i is chosen to be th maximum of the 
'''
def optimal_w_choices(x,sigma,x_cor,k):
    
    dim = x.shape[0]
    Y = np.zeros(k)
    W = np.zeros([k,dim])
    for i in range(k):
        W_restricted = W[:i]
        
        #Calculating the new Correlatoin matrix
        if i > 0:
            Inv = np.linalg.inv( W_restricted.dot(x_cor).dot(W_restricted.T) + np.identity(i)*sigma*sigma)
            New_cor = x_cor - x_cor.dot(W_restricted.T).dot( Inv ).dot(W_restricted.dot(x_cor.T))
        else:
            New_cor = x_cor
            
        # Search foe the best w_i
        W[i] = max_eigen_value(New_cor)
        Y[i] = W[i].dot(x) + np.random.normal()*sigma*sigma
    
    #Sanity check Estimating X by least square
#     WT_W = W.T.dot(W)
#     WT_Y = W.T.dot(Y)
    Temp1 = W.dot(x_cor).dot(W.T) + np.identity(k)*sigma*sigma
    X_estimate = x_cor.dot(W.T).dot( np.linalg.inv(Temp1) ).dot(Y)
    
    return np.linalg.norm(X_estimate-x)

def random_w_choices(x,sigma,x_cor,k):
    dim = x.shape[0]
    Y_random = np.zeros(k)
    Y_optimal = np.zeros(k)
    W_random = np.zeros([k,dim])
    W_optimal = np.zeros([k,dim])
    for i in range(k):
        #Picking w_i randomly
        W_random[i] = random_unit_norm(dim, 1000) #Large value of sigma for improving the unformity
        
        #picking w_i optimally
        W_restricted = W_optimal[:i]
        #Calculating the new Correlatoin matrix
        if i > 0:
            Inv = np.linalg.inv( W_restricted.dot(x_cor).dot(W_restricted.T) + np.identity(i)*sigma*sigma)
            New_cor = x_cor - x_cor.dot(W_restricted.T).dot( Inv ).dot(W_restricted.dot(x_cor.T))
        else:
            New_cor = x_cor        
        # Search foe the best w_i
        W_optimal[i] = max_eigen_value(New_cor)
        
        noise = np.random.normal()*sigma*sigma
        Y_random[i] = W_random[i].dot(x) + noise
        Y_optimal[i] = W_optimal[i].dot(x) + noise
    
    #Sanity check Estimating X by least square
#     WT_W = W_random.T.dot(W_random)
#     WT_Y = W_random.T.dot(Y_random)
    #X_LS_est = np.linalg.inv(WT_W).dot(WT_Y)
    
    Temp1 = W_random.dot(x_cor).dot(W_random.T) + np.identity(k)*sigma*sigma
    X_random_estimate = x_cor.dot(W_random.T).dot( np.linalg.inv(Temp1) ).dot(Y_random)
    
    Temp2 = W_optimal.dot(x_cor).dot(W_optimal.T) + np.identity(k)*sigma*sigma
    X_optimal_estimate = x_cor.dot(W_optimal.T).dot( np.linalg.inv(Temp2) ).dot(Y_optimal)

#     print "Least squares : Error fraction" , np.linalg.norm(X_LS_est-x)/np.linalg.norm(x)
#     print "Random : Error fraction" , np.linalg.norm(X_estimate-x)/np.linalg.norm(x)
    return np.linalg.norm(X_random_estimate-x) , np.linalg.norm(X_optimal_estimate-x) , 

def calc(x_cor,x_mean):
        
    dim = np.shape(x_cor)[0];
    num_tests = 400;
    sigma = 20;
    K_vals = [2, 4, 8, 16,32,64,128,256,];
    Random_error = np.zeros([len(K_vals)])
    Optimal_error = np.zeros([len(K_vals)])
#     TOptimal_error = np.zeros([len(K_vals)])
    
    x = np.random.multivariate_normal(x_mean, x_cor)
    for j, k in enumerate( K_vals ):
        print k
        print Random_error
        print Optimal_error, "\n"
        #print TOptimal_error, "\n"
        for i in range(num_tests):
            if i%25 == 0:
                print i
            R,O = random_w_choices(x, sigma, x_cor,k)
            Random_error[j] = +R/num_tests
            Optimal_error[j] = +O/num_tests
            #TOptimal_error[j] = +optimal_w_choices(x, sigma, x_cor,k)/num_tests
            
    print Random_error
    print Optimal_error
    
#     plt.semilogy(K_vals,Random_error, marker='o', lw=0.5, color="blue", basey=2,alpha=1)
#     plt.semilogy(K_vals,Optimal_error, marker='o', lw=0.5, color="red", basey=2,alpha=1)
    return (K_vals,Optimal_error/Random_error)

def main():    
    x_mean = np.zeros([10])
    x_cor = np.array([
     [5.279, 3.616, 3.651, 3.127, 3.877, 3.025, 2.48,  3.548, 3.019, 3.374],
     [3.616, 3.837, 2.704, 2.119, 2.958, 2.787, 2.006, 2.898, 1.932, 2.068],
     [3.651, 2.704, 3.546, 2.216, 2.681, 2.694, 1.882, 2.604, 2.006, 2.856],
     [3.127, 2.119, 2.216, 2.401, 2.423, 2.101, 1.654, 2.535, 2.239, 2.112],
     [3.877, 2.958, 2.681, 2.423, 3.72,  3.032, 2.47,  2.536, 2.517, 2.35 ],
     [3.025, 2.787, 2.694, 2.101, 3.032, 3.513, 2.377, 2.843, 2.139, 2.491],
     [2.48,  2.006, 1.882, 1.654, 2.47,  2.377, 1.893, 1.995, 1.9,   1.903],
     [3.548, 2.898, 2.604, 2.535, 2.536, 2.843, 1.995, 4.154, 2.742, 2.569],
     [3.019, 1.932, 2.006, 2.239, 2.517, 2.139, 1.9,  2.742, 3.174, 2.127],
     [3.374, 2.068, 2.856, 2.112, 2.35,  2.491, 1.903, 2.569, 2.127, 3.165],
    ]);
     
    Xaxis, Yaxis = calc(x_cor,x_mean)
    plt.plot(Xaxis, Yaxis,marker='o', lw=0.5, color="blue",alpha=1)

    x_iden = np.identity(10)
    Xaxis, Yaxis = calc(x_iden,x_mean)
    plt.plot(Xaxis, Yaxis,marker='o', lw=0.5, color="red",alpha=1)
        
    red_patch =  mpatches.Patch(color='red', label='Indentity')
    blue_patch = mpatches.Patch(color='blue', label='Correlated')
    plt.legend(handles=[red_patch,blue_patch])
    plt.show()
    
    
main()