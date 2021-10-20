import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
import scipy
import time
from sklearn.metrics import confusion_matrix
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html



# This function is to calculate the W based on the LMS algorithm
# For this kind of approach, there should be no w0.
def calculate_W(X, Y, lam):
    [a, b] = np.shape(X)

    #     w=np.zeros((b,1))
    temp = np.linalg.pinv(lam * np.eye(b) + np.dot(X.T, X))
    temp2 = np.dot(temp, X.T)
    w = np.dot(temp2, Y)

    return w

# This function is to give the matrix of good weight decay
def loglam(m,X,L):
    u,s,v=np.linalg.svd(X)
    lmax=np.sum(s)
#     print(lmax)
    lmin=np.min(s)
#     print(lmin)
    loglam1=np.zeros(L)
    # for k in range(L):
    #     loglam1[k]=np.log(lmin)+(k)/(L-1)*(np.log(lmax)-np.log(lmin))
    return loglam1


# This function is to normalize the power output

def Eb(lamb, T):
    c1 = 3.74e-16;
    #     %constant C1, unit W*m^2
    c2 = 1.4388e-2;
    #     %constant C2, unit m*K
    temp1 = c1 * lamb ** (-5);
    temp2 = np.exp(c2 / lamb / T) - 1;

    Eb1 = temp1 / temp2;
    return Eb1


# This function is to disctritize the frequency space in N resolution
def space(n, arrays_properties):
    m, N = arrays_properties.shape  # m is the number of samples we have
    #     print(N)
    fre = np.arange(0, N, 1)
    #     fre=np.reshape(fre,(N,1))
    fre_new = np.arange(0, N, N / n)
    #     fre_new=np.reshape(fre_new, (n,1))
    f = interp1d(fre, arrays_properties, axis=1)
    new_array = f(fre_new)

    return new_array


def get_arrayproperties ():
    # Important import filter spectra
    with open('filter/filter_20.4.csv', 'r', encoding='utf-8-sig') as f:
        Spectra = np.genfromtxt(f, dtype=float, delimiter=',', skip_header=False)
    #Spectra = np.genfromtxt('filter_100.csv', \
                           # delimiter=',', skip_header=False)

    # N=1001
    # D=40
    # width=50
    #
    # # plt.figure()
    # Spectra =np.zeros ((N,D))
    # position=np.arange (0, N, 1 )
    # for idx in range(D):
    #     temperary = norm.pdf(position, N / (D - 1) * idx, width)
    #     Spectra [:, idx] = temperary / (temperary.max())

    (N,m)=Spectra.shape
    print (N,m)
    # Important frequency starting point
    starting_frequency=450.
    ending_frequency=3970.

    axis_nm = np.arange( starting_frequency,ending_frequency, (ending_frequency-starting_frequency)/N)
    print (axis_nm.shape)

    # plt.figure()
    # for i in range(m):
    #     plt.plot(axis_nm, Spectra[:, i])
    #
    # plt.xlabel('wavelength (nm)')
    # plt.ylabel('Transmission')
    # plt.show()
    arrays_properties = Spectra.T


    return arrays_properties, axis_nm


def generate_signal_of_interest_from_florou (filter_properties, S_matrix):

    (m,N)=np.shape (filter_properties)
    (N,d) = np.shape(S_matrix)
    # floro=np.random.random((d,1))
    floro=np.zeros((d,1))
    flo_index=np.random.randint(high=d, low=0)

    floro[flo_index,0]=np.random.random()+1.


    flo_id=flo_index

    spectra_GT=np.dot(S_matrix,  floro)
    signal_GT = np.dot(filter_properties, spectra_GT)


    angle=np.arange(0,m)
    sigma1 = 0.1
    signal_noisy = signal_GT *(1+ np.random.normal(0, sigma1, (m, 1)))
    signal_noisy = np.round(signal_noisy, 6)

    # plt.figure()
    #
    # plt.plot(angle,signal_GT, label='ground truth')
    # plt.plot(angle, signal_noisy, label='noisy data')
    # plt.legend()
    #
    # plt.xlabel('array index')
    # plt.ylabel('power_read (W)')
    # plt.show()
    # #
    # plt.figure()
    # for i in range(d):
    #     plt.plot(S_matrix[:, i])
    #     plt.legend(str (i))
    #
    #
    # plt.xlabel('wavelength (nm)')
    # plt.ylabel('Transmission')
    # plt.show()

    return signal_GT, signal_noisy, floro,flo_id, spectra_GT



def transforming_spectra ( S_matrix1, flo1):

    (N,D)=np.shape(S_matrix1)
    S_matrix3=np.zeros((N,1))
    S_matrix3[:,0]=S_matrix1[:,flo1]


    return S_matrix3


def construct_for_scipy (X, Y,lam):

    (m, D)=np.shape (X)
    I_matrix=np.eye (D) * lam
    C_matrix=np.concatenate((X,I_matrix))
    zeros=np.zeros((D,1))
    Y_vector=np.concatenate((Y, zeros))

    Y_vector=Y_vector.flatten()
    # print ('X',X, 'C_matrix',C_matrix)

    return C_matrix, Y_vector


# The Math overall is : X(SW)=(XS)W=Y, where X is the filter arrays, m *N, S is N* D, W is D*1, Y is m*1.
# S is the matrix defining how wide the spectra is, each N*1 is a gassusian shape with FWHM w
# SW is the real spectra

def calculate_spectra_at_one_point(array_properties, signal_noisy,axis_cm, Spectra_groud_truth, signal_ground_truth,floro_GT, S_matrix_load, flo_id):
    # Important Hyperparameter choice
    (dump,number_of_floro_possible1)=np.shape (S_matrix_load)

    number_of_floro_possible2 =number_of_floro_possible1

    L=1
    # Important NXF is to loop the sample, tunable hyperparameters
    nxf=3

    # Width_choice = [5, 10]
    # Width_choice is the width/N*bandwidth

    (m,N_array)=np.shape (array_properties)
    errors=np.zeros ((number_of_floro_possible1, L))

    # Loop 1, find right dimension D
    for flo1 in range ( number_of_floro_possible1):
        dimension=1


        # Loop 2, find the width term of S matrix

        S_matrix = transforming_spectra(S_matrix_load, flo1)
        X_matrix = np.dot(array_properties, S_matrix)
        # weight_inital_guess=calculate_W(X=X_matrix, Y=signal_noisy, lam=0)
        logLambda = loglam(m, X_matrix, L)
        # logLambda=np.zeros((L,))
        #             Loop3, find the right right regularization parameter
        for ell in range(L):
            lam = np.exp(logLambda[ell])
            # lam=0

            error_temperary = 0

            for xfvi in range(nxf):
                X_train = []
                Y_train = []
                X_test = []
                Y_test = []
                # Set up the training and testing arrays
                for j in range(m):
                    if j % nxf == xfvi:
                        X_test = np.append(X_test, X_matrix[j, :])
                        Y_test = np.append(Y_test, signal_noisy[j])



                    else:
                        X_train = np.append(X_train, X_matrix[j, :])

                        Y_train = np.append(Y_train, signal_noisy[j])

                X_train = np.array(X_train)
                X_test = np.array(X_test)
                Y_train = np.array(Y_train)
                Y_test = np.array(Y_test)

                X_train = X_train.reshape(int(np.size(X_train) / (dimension)), dimension)
                Y_train = Y_train.reshape(int(np.size(X_train) / (dimension)), 1)
                X_test = X_test.reshape(int(np.size(X_test) / (dimension)), dimension)
                Y_test = Y_test.reshape(int(np.size(X_test) / (dimension)), )

                [M2, iii] = np.shape(X_test)

                (X_train_for_scipy, Y_train_for_scipy) = construct_for_scipy(X_train, Y_train, lam)
                #
                w, dump = scipy.optimize.nnls(X_train_for_scipy, Y_train_for_scipy)

                est1 = np.dot(X_test, w)

                tDiff = (Y_test - est1)

                error_temperary = error_temperary + np.sum(tDiff * tDiff) / (M2)
                # print (error_temperary, 'error temp')


            errors[flo1, ell] = error_temperary


    # print(errors)
    idx_optimal1=np.where(errors==np.min(errors))
    # print(idx_optimal1)


    idx_optimal=idx_optimal1

    # print (idx_optimal1, 'final used')

    if np.size (idx_optimal[0])>1:
        flo1_choice = int((idx_optimal[0])[0])

        Regularization_opt = np.exp(logLambda[int((idx_optimal[1])[0])])
    else:
        flo1_choice = int(idx_optimal[0])

        Regularization_opt = np.exp(logLambda[int(idx_optimal[1])])

    S_matrix_opt = transforming_spectra(S_matrix_load,flo1_choice)
    X_matrix_opt = np.dot(array_properties, S_matrix_opt)

    (X_matrix_opt_1,signal_noisy_1) = construct_for_scipy(X_matrix_opt, signal_noisy, Regularization_opt)

    final_w, dump=scipy.optimize.nnls(X_matrix_opt_1,signal_noisy_1)
    # final_w, dump = scipy.optimize.nnls(X_matrix_opt, signal_noisy)
    signal_calculated=np.dot(X_matrix_opt,final_w)

    Regressed_spectra=np.dot (S_matrix_opt,final_w)

    flo_id_calculated=np.sort([flo1_choice])
    flo_intensity=np.zeros((number_of_floro_possible1,1))
    flo_intensity[flo1_choice]=final_w[0]


    # print (dimension_opt, width_opt, Regularization_opt, idx_optimal)


    # plt.figure()
    #
    # plt.plot(axis_cm, Spectra_groud_truth/ (Spectra_groud_truth.max()), label='ground truth')
    # plt.plot(axis_cm, Regressed_spectra/(Regressed_spectra.max()), label='Regressed Spectra')
    # plt.legend()
    #
    # plt.xlabel('wavelength (nm)')
    # plt.ylabel('Normalized intensity')
    # plt.title ('spectrum')
    # # plt.show()
    #
    # plt.figure()
    #
    # plt.plot(signal_calculated[0:m], label='Regressed signal ')
    # plt.plot(signal_noisy, label='Measured signal (including noise) ')
    # plt.plot(signal_ground_truth, label='ground truth')
    # plt.legend()
    #
    # plt.xlabel('array number')
    # plt.ylabel('power_read (W)')
    # plt.title('signal regression')
    # # plt.show()
    #
    # plt.figure()
    #
    # plt.scatter([dimension_opt,width_opt ],final_w, label='regressed floro ', c='#ff7f0e')
    # plt.plot(floro_GT, label='ground truth')
    # plt.legend()
    #
    # plt.xlabel('floro index')
    # plt.ylabel('floro intensity')
    # plt.title('floro intensity'+ str(final_w))
    # plt.show()

    return flo_id_calculated,flo_intensity


(filter_properties, axis_cm)=get_arrayproperties()
with open('spetrum/transmittance.csv', 'r', encoding='utf-8-sig') as f:
    S_matrix1_load = np.genfromtxt(f, dtype=float, delimiter=',', skip_header=False)

#S_matrix1_load = np.genfromtxt('absorptance.csv', \
                          #delimiter=',', skip_header=False)
# Spectra_GT=generate_spectra_of_interest( axis_cm=axis_cm)
(N,D)=np.shape (S_matrix1_load )

total_test=1000
flo_id_GT=np.zeros((total_test,))
flo_id_pred=np.zeros((total_test,))
flo_intensity_GT=np.zeros((total_test,D))
flo_intensity_pred=np.zeros((total_test,D))
start=time.time()

for idx in range (total_test):
    (signal_GT, signal_noisy, floro_GT,flo_id, Spectra_GT) = generate_signal_of_interest_from_florou(
        filter_properties=filter_properties, S_matrix=S_matrix1_load)


    flo_id_calculated, flo_intensity=calculate_spectra_at_one_point(array_properties=filter_properties, signal_noisy=signal_noisy, axis_cm=axis_cm,
                                   Spectra_groud_truth=Spectra_GT, signal_ground_truth=signal_GT, floro_GT=floro_GT, S_matrix_load=S_matrix1_load,flo_id=flo_id)
    cycles=idx

    flo_id_GT[idx:idx+1]=flo_id
    flo_id_pred[idx:idx+1]= flo_id_calculated
    flo_intensity_GT[idx, :]=np.reshape(floro_GT, (D,))
    flo_intensity_pred[idx, :]=np.reshape(flo_intensity, (D,))



end=time.time()
print (end-start,'seconds', 'operating time for', cycles+1, 'pixels')

confusion_result=confusion_matrix( flo_id_GT, flo_id_pred)
print(confusion_result)

epsilon_offset=1e-7

dia=np.zeros((total_test,D))
dia[flo_intensity_GT>0]=1

dif=(flo_intensity_GT-flo_intensity_pred)*dia/(flo_intensity_GT+epsilon_offset)
dif_MSE=np.abs(dif)
dif_MSE=np.sum (dif_MSE, axis=0)/np.sum(dia, axis=0)
print (dif_MSE*100., 'percent of error of intensities for each types')

error=np.sum(dif_MSE)/D*100
print('The overall error of intensities is',error,'percent')
print('The overall accuracy is',100-error,'percent')