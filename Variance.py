import pandas as pd
import numpy as np
from scipy.linalg import svd
import csv
from sklearn.decomposition import TruncatedSVD
with open("tao's project/Code and data(TAO)/data/fluorophore_T.csv", 'r', encoding='utf-8-sig') as f:
#with open("spetrum/transmittance_AT.csv", 'r', encoding='utf-8-sig') as f:
    A = np.genfromtxt(f, dtype=float, delimiter=',', skip_header=False)
    wavenumber = []
    wavenum = 500 #450
    while wavenum <=800: #< 3970:
        wavenumber.append(wavenum)
        wavenum += 0.3 #4
        wavenum=round(wavenum,1)
    array = pd.DataFrame(data=A,
                       columns=wavenumber)
    array.loc['var'] = array.var(0)
    print(array)

  #print(svd)
    array_sorted = array.sort_values(by='var',key=abs, axis=1,ascending=False)
    print(array_sorted)
    wav_sorted = array_sorted.columns.values
    print(wav_sorted)

    spectrum = pd.DataFrame(data=A,
                        columns=wavenumber)
    print(spectrum)

    filter_rec = [wav_sorted[0]]
    print(filter_rec)
    for current in wav_sorted:
        independent= True
        for picked in filter_rec:
            if(abs(spectrum[current].corr(spectrum[picked])))>0.9:
                independent = False

        if independent:
            filter_rec.append(current)


    print(filter_rec)
    print(len(filter_rec))
    #with open('var/filter_rec_a.csv', 'w', newline='') as file:
    with open("tao's project/Code and data(TAO)/filter/var/filter_rec.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([filter_rec])
