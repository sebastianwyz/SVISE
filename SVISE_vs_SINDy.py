from functools import partial
import matplotlib.pyplot as plt
from matplotlib import rc
import torch
import pysindy as ps
from torch.utils.data import DataLoader, TensorDataset
from torchsde import sdeint as torchsdeint
from tqdm import tqdm
from svise.sde_learning import *
from svise.variationalsparsebayes.sparse_glm import SparsePolynomialNeighbour1D
from svise.variationalsparsebayes.sparse_glm import SparsePolynomialSinusoidTfm
from svise import quadrature
from svise import odes
from svise import sde_learning, sdeint
import torch
import torch.optim.lr_scheduler as lr_scheduler
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time
from time import perf_counter as pfc
import os
import pathlib
import argparse
import networkx as nx
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import inferno
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import re
import numpy as np
from torch import threshold
from svise import sde_learning, sdeint
from matplotlib import rc
from tqdm import tqdm
from svise.sde_learning import SparsePolynomialSDE
from svise import quadrature
from svise import odes
from svise.utils import solve_least_squares
import os
import pathlib
import argparse
import networkx as nx

from sklearn.preprocessing import StandardScaler

from svise import sde_learning, sdeint

folder_path = "./SVISE_vs_SINDy/CORRETTO/MEGA CORRETTO/ER/03"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
rc('text', usetex=False)
torch.set_default_dtype(torch.float64)

N = 20  # Numero di nodi
n = 2   # Numero di gruppi
p=0.2
# Assumendo che vuoi due gruppi di dimensione uguale
sizes = [N // n] * n  # Crea una lista con due gruppi di dimensione 10



# Crea il modello di blocco stocastico
#SW = nx.stochastic_block_model(sizes, p)
radius = 0.35  # Raggio di connessione

SW = nx.random_geometric_graph(N, radius)

# Converti in array numpy
#A = nx.to_numpy_array(SW)




# Crea il grafo geometrico casuale
A = nx.random_geometric_graph(N, radius)


A = np.loadtxt("C:/Users/teresa i robert/Desktop/Tesi/svise-main/svise-main/experiments/8_tutorial_preda_cacciatore/RetiNeurali/SVISE_vs_SINDy/CORRETTO/MEGA CORRETTO/ER/matrice_er.txt")

file_path = os.path.join(folder_path, "matrice_er.txt")

# Salva la matrice in un file di testo
np.savetxt(file_path, A, fmt='%d')  # Usa '%.4f' se vuoi salvare con 4 cifre decimali


#A = torch.tensor(A, dtype=torch.float32)  
############# BA ##################
#A = np.loadtxt("C:/Users/teresa i robert/Desktop/Tesi/svise-main/svise-main/experiments/8_tutorial_preda_cacciatore/RetiNeurali/BA.adj")
#A = np.loadtxt("/home/ubuntu/svise-main/experiments/8_tutorial_preda_cacciatore/RetiNeurali/SBM.adj")

############# SW ##################
#A = np.loadtxt("C:/Users/teresa i robert/Desktop/Tesi/svise-main/svise-main/experiments/8_tutorial_preda_cacciatore/RetiNeurali/SW.adj")
############# SBM #################
#A = np.loadtxt("C:/Users/teresa i robert/Desktop/Tesi/svise-main/svise-main/experiments/8_tutorial_preda_cacciatore/RetiNeurali/SBM.adj")
############# GRG #################
#A = np.loadtxt("C:/Users/teresa i robert/Desktop/Tesi/svise-main/svise-main/experiments/8_tutorial_preda_cacciatore/RetiNeurali/GRG.adj")
#A=nx.to_numpy_array(SW)

plt.figure(figsize=(3, 3))  
plt.imshow(A, cmap='inferno')  
plt.colorbar()  
plt.title('Matrice A')
plt.xlabel('Indice ')
file_path = f"{folder_path}/matrice A.png"
plt.savefig(file_path)

plt.close()

G = torch.zeros(N,N) + torch.eye(N)

torch.manual_seed(0)

def lorenz(sigma, rho, beta, t, x):
    
    dtheta = torch.zeros_like(x) 
    Ku=0.3
    u = [           # I used this set of proper frequencies for every case
    0.5539, -0.3248, -0.8373, 
    0.2874, -0.4956, 0.6284, -0.7612,
    0.4321, -0.2134, 0.9876, -0.6543,
    0.1212, -0.8989, 0.3344, 0.4567,
    -0.7890, 0.6789, -0.2345, 0.5678,
    -0.3456, 0.7891, -0.1234, 0.4321,
    0.3621,-0.5471, 0.7362, 0.4102,
    -0.5678, 0.3210, -0.6543, 0.1098,
    -0.8765, 0.5432, -0.3210, 0.4560
    ]
    w=[0.0]*N
    for i in range(N):
        w[i] = 5*u[i]

    for i in range(N):
        dtheta[...,i] = w[i] + Ku * sum(A[i, j] * torch.sin(x[...,j] - x[...,i]) for j in range(N) if i != j)
    return dtheta

        

class Lorenz:  # nothing to do with lorenz
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self) -> None:
        self.ode = partial(lorenz, 10, 28, 8/3)
        self.diff = torch.ones(1, N) * 1e-3

    def f(self, t, y):
        return self.ode(t, y)

    def g(self, t, y):

        return self.diff

tempo = [8]  
rms_means = []
sigma_means = []
rms_means_sindy = []
sigma_means_sindy = []
rms_means_1 = []
sigma_means_1 = []
rms_means_sindy_1 = []
sigma_means_sindy_1 = []
for tempo in tempo:
    SNR = [100]
    total_sum_final_values_10 = []
    total_sum_final_values_20 = []
    total_sum_final_values_100 = []
    total_sum_finale_values_10 = []
    total_sum_finale_values_20 = []
    total_sum_finale_values_100 = []
    total_sum_final_values_10_sindy = []
    total_sum_final_values_20_sindy = []
    total_sum_final_values_100_sindy = []
    total_sum_finale_values_10_sindy = []
    total_sum_finale_values_20_sindy = []
    total_sum_finale_values_100_sindy = []
    total_sum_final_values_valori = []
    total_sum_finale_values_valori = []
    total_sum_final_values_sindy_valori = []
    total_sum_finale_values_sindy_valori = []
    total_sum_final_values_10_valori = []
    total_sum_final_values_20_valori = []
    total_sum_final_values_100_valori = []
    total_sum_finale_values_10_valori = []
    total_sum_finale_values_20_valori = []
    total_sum_finale_values_100_valori = []
    total_sum_final_values_10_sindy_valori = []
    total_sum_final_values_20_sindy_valori = []
    total_sum_final_values_100_sindy_valori = []
    total_sum_finale_values_10_sindy_valori = []
    total_sum_finale_values_20_sindy_valori = []
    total_sum_finale_values_100_sindy_valori = []
    for SNR in SNR:
        def dizionario(n_data: int, tend: float, s:float) -> dict:
            x00 = [
            0.5539, 0.3248, 0.8373, 
            0.2874, 0.4956, 0.6284, -0.7612,
            0.4321, -0.2134, 0.9876, -0.6543,
            0.1212, -0.8989, 0.3344, 0.4567,
            -0.7890, 0.6789, -0.2345, 0.5678,
            -0.3456, 0.7891, -0.1234, 0.4321,
            0.3621,-0.5471, 0.7362, 0.4102,
            -0.5678, 0.3210, -0.6543, 0.1098,
            -0.8765, 0.5432, -0.3210, 0.4560
            ]
            x0=[0.0]*(N)
            for i in range(N):
                x0[i] = 0*x00[i]
            t_span = [0, tend + s]
            t_eval = torch.linspace(t_span[0], t_span[1], n_data)
            x0 = torch.as_tensor(x0, dtype=torch.float32).unsqueeze(0)
            dt = t_eval[1] - t_eval[0]
            sde_kwargs = dict(dt=dt, atol=1e-5, rtol=1e-5, adaptive=True)
            sol = torchsdeint(Lorenz(), x0, t_eval, **sde_kwargs).squeeze(1)
            potenza_segnale = torch.var(sol)
            SNR_lineare = 10 ** (SNR / 10)
            potenza_rumore = potenza_segnale / SNR_lineare
            std = torch.sqrt(potenza_rumore)
        
            train_ind = t_eval <= tend 
            data = dict(t=t_eval, true_state=sol)
            dtheta_dt = lorenz(10, 28, 8/3, t_eval, sol)
            data["std"] = std
            data["y"] = sol @ G.T
            data["y"] += torch.randn_like(data["y"]) * std
            data["train_t"] = data["t"][train_ind]
            data["train_y"] = data["y"][train_ind]
            data["valid_t"] = data["t"][~train_ind]
            data["valid_state"] = data["true_state"][~train_ind]
            data["train_state"] = data["true_state"][train_ind]
            data["fase"] = dtheta_dt 
            data["fase_train"] = data["fase"][train_ind]
            data["fase_val"] = data["fase"][~train_ind]

            return data

        if tempo ==4:
            m=1
            num_dati = 256
        if tempo ==8:
            m=2
            num_dati = 512 
        if tempo ==16:
            m=4
            num_dati = 1024
        data = dizionario(num_dati , tempo, m)
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(data["train_t"].numpy(), data["train_y"].numpy(), "k.", label="training data", alpha=0.3)
        ax[0].set_ylabel("observations")
        ax[1].plot(data["t"].numpy(), data["true_state"].numpy(), 'b-', label='latent state')
        ax[1].set_ylabel("latent state")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = f"{folder_path}/true_state_{tempo}_{num_dati}_{SNR}.png"

        plt.savefig(file_path)
        combined_sin_labels = []
        input_labels = []

        for i in range(1, N + 1):
            input_labels.append(f"theta_{i}")

        for i in range(N):
            for j in range(i  , N):
                if i!=j: 
                    combined_sin_labels.append(f"sin({input_labels[i]} - {input_labels[j]})")
        #input_labels += combined_sin_labels 
        input_labels = combined_sin_labels


        correct_terms_strings = []
        for i in range(A.shape[0]):
            terms = set()
            for j in range(A.shape[1]):
                if A[i, j] == 1 and i < j:  
                    terms.add(f"sin(theta_{i+1} - theta_{j+1})")
                elif A[i, j] == 1 and i > j:  
                    terms.add(f"sin(theta_{j+1} - theta_{i+1})")
            correct_terms_strings.append(terms)
            print("correct terms for node:", i ,correct_terms_strings[i])


        t_span = (data["train_t"].min(), data["train_t"].max())
        d = N 
        degree = 1 
        n_reparam_samples = 128 #len(data["train_t"])  # how many reparam samples
        var = (torch.ones(d) * data["std"]) ** 2
        num_data = len(data["train_t"])

        sde = SparsePolynomialSDE(
            d, 
            t_span, 
            degree=degree, 
            n_reparam_samples=n_reparam_samples, 
            G=G, 
            num_meas=d, 
            measurement_noise=var,
            train_t=data["train_t"], 
            train_x=data["train_y"],
            input_labels=input_labels
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        sde.to(device)
        sde.train()  # back in train mode

        batch_size = int(min(128, len(data["train_t"])))
        #batch_size = len(data["train_t"])  
        num_iters = 2000
        transition_iters = 500
        num_mc_samples = int(min(128, len(data["train_t"])))
        summary_freq = 100
        scheduler_freq = transition_iters // 2
        lr = 1e-2

        optimizer = torch.optim.Adam(
            [{"params": sde.state_params()}, {"params": sde.sde_params(), "lr": 1e-2}], lr=lr
        )


        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        start_time = time.time()
        train_dataset = TensorDataset(data["train_t"], data["train_y"])
        train_loader = DataLoader(train_dataset, batch_size=num_mc_samples, shuffle=True)
        num_epochs = num_iters // len(train_loader)
        j = 0

        t = data["train_t"].to(device)
        y_data = data["train_y"].to(device)
        t0 = 0
        tf = tempo
        t_eval = torch.linspace(t0, tf, int(num_dati * 4 / 5)).to(device)
        log_likelihood_threshold = 50

        for epoch in range(num_epochs):
            for t_batch, y_batch in train_loader:
                t_batch = t_batch.to(device)
                y_batch = y_batch.to(device)
                j += 1
                optimizer.zero_grad()
                idx = np.random.choice(np.arange(num_data), num_mc_samples, replace=False)
                if j < (transition_iters):
                    # beta warmup iters
                    beta = min(1.0, (1.0 * j) / (transition_iters))
                    train_mode = "beta"
                else:
                    beta = 1.0
                    train_mode = "full"
                if j % scheduler_freq == 0:
                    scheduler.step()
                if j % summary_freq == 0:
                    print_loss = True
                else:
                    print_loss = False
                loss = -sde.elbo(t[idx], y_data[idx], beta, num_data, print_loss=print_loss)
                #loss.backward(retain_graph=True)
                loss.backward()

                optimizer.step()
                if j % summary_freq == 0:
                    sde.eval()
                    mu = sde.marginal_sde.mean(t_eval)
                    covar = sde.marginal_sde.K(t_eval)
                    var = covar.diagonal(dim1=-2, dim2=-1)
                    print(
                        f"iter: {j:05}/{num_iters:05} | loss: {loss.item():04.2f} | mode: {train_mode} | time: {time.time() - start_time:.2f} | beta: {beta:.2f} | lr: {scheduler.get_last_lr()[0]:.5f}",
                        flush=True,
                    )
                    sde.sde_prior.update_sparse_index()
                    print_str = f"Learnt governing equations: {sde.sde_prior.feature_names[0]}"
                    #print("correct terms for node:", 0 ,correct_terms_strings[0])
                    #print(print_str)
                    sde.sde_prior.reset_sparse_index()
                    start_time = time.time()
                    lb = mu - 2 * var.sqrt()
                    ub = mu + 2 * var.sqrt()
                    sde.train()  # back in train mode
                

        sde.eval()
        #sde.resample_sde_params()
        sde.sde_prior.reset_sparse_index()
        sde.sde_prior.update_sparse_index()
        torch.cuda.empty_cache()
        variabili = []
        sde_prior_feature_names = []


        for j, eq in enumerate(sde.sde_prior.feature_names):
            print(f"'{eq}',")
            sde_prior_feature_names.append(eq)

        sde_prior_feature_names = [eq for eq in sde.sde_prior.feature_names]




        correct_terms_strings = []
        for i in range(A.shape[0]):
            terms = set()
            for j in range(A.shape[1]):
                if A[i, j] == 1 and i < j:  # Check only one connection per pair (i < j)
                    terms.add(f"sin(theta_{i+1} - theta_{j+1})")
                elif A[i, j] == 1 and i > j:  # Add the inverse term if not considered in i < j
                    terms.add(f"sin(theta_{j+1} - theta_{i+1})")
            correct_terms_strings.append(terms)
            print("correct terms for node:", i ,correct_terms_strings[i])





        sde.eval()

        num_nodi = N
        RMS_1 = []
        sigmaRMS_1= []
        ########################################## forward 1s ##################################################
        state_truee = dizionario(num_dati, tempo, 1)
        ######################################################      #SINDY    #############################################################################################
        '''
        def sin_diff(x, y):
            return np.sin(x - y)

        # Creazione della libreria personalizzata con funzioni specifiche
        custom_functions = [sin_diff]
        custom_function_names = [lambda x, y: f"sin({x} - {y})"]

        custom_library = ps.CustomLibrary(
            library_functions=custom_functions,
            function_names=custom_function_names
        )

        # Creazione della libreria di polinomi di grado 0
        polynomial_library = ps.PolynomialLibrary(degree=0)

        # Combinazione delle librerie
        combined_library = ps.GeneralizedLibrary([custom_library, polynomial_library])

        # Assumiamo che 'data' sia un dizionario contenente i dati di addestramento
        x_train = state_truee["train_y"].detach().cpu().numpy()
        dt = state_truee["train_t"].detach().cpu().numpy()
        
        #scaler = StandardScaler()
        #x_train_scaled = scaler.fit_transform(x_train)
        ##optimizer = ps.STLSQ(threshold=0.05, alpha=0.2, max_iter=100000)
        optimizer = ps.STLSQ(threshold=0.05, alpha= 0.2, max_iter = 100000)
        #optimizer = ps.STLSQ(threshold=0.06, alpha=0.01, max_iter=100000)

        # Inizializzazione e adattamento del modello SINDy con il nuovo ottimizzatore
        model = ps.SINDy(feature_library=combined_library, optimizer=optimizer)
        model.fit(x_train, t=dt)
        #model.print()
        # Opzionalmente, predizioni del modello
        predictions = model.predict(x_train)
        predictions = model.differentiate(x_train, t=dt)
        #print("tipo di prediction",type(predictions))



        #x_train = state_truee["valid_state"].detach().cpu().numpy()
        x0 = x_train[-1, :]  # Condizioni iniziali (primo punto di dati) ############################### invece di 0

        #predictions = model.simulate(x0, state_truee["valid_t"])
        #predictions = model.simulate(x0, dt)



        y0 = predictions[-1, :]  # Condizioni iniziali (primo punto di dati)
        #print(x0)
        dtt = state_truee["valid_t"].detach().cpu().numpy()
        #simulated_data = model.simulate(y0, dtt)
        '''
        
        ####################################################################################################################################################
        K=32        #num_dati
        with torch.no_grad():
            # get the state estimate
            mu = sde.marginal_sde.mean(state_truee["train_t"])
            var = sde.marginal_sde.K(state_truee["train_t"]).diagonal(dim1=-2, dim2=-1)
            lb = mu - 2 * var.sqrt()
            ub = mu + 2 * var.sqrt()

            # generate a forecast using 32 samples
            x0 = mu[-1] + torch.randn(num_mc_samples, N) * var[-1].sqrt() ################# 30????????????????????????????''(len(data["train_t"]
            sde_kwargs = dict(dt=1e-2, atol=1e-2, rtol=1e-2, adaptive=True)
            t_eval = torch.linspace(state_truee["train_t"].max(), state_truee["t"].max(), len(state_truee["valid_state"]))  #38.4 x 4
            xs = sdeint.solve_sde(sde, x0, t_eval, **sde_kwargs)  
            pred_mean_1 = xs.mean(1)
            pred_lb = pred_mean_1 - 2 * xs.std(1) 
            pred_ub = pred_mean_1 + 2 * xs.std(1) 


        fig, axs = plt.subplots(2,1, figsize=(6,7))
        ###axs[1].plot(state_truee["train_t"].numpy(), predictions ,color="C2")
        ###axs[1].plot(state_truee["valid_t"].numpy(), simulated_data ,color="C2")

        for i in range(1):
            axs[0].plot(state_truee["train_t"].numpy(), mu.numpy()[:, i], 'C0', label='state estimate' if i == 0 else "_nolegend_")

        ###axs[0].plot(state_truee["valid_t"].numpy(), simulated_data[:, 0] , color="#8B0000", label='SINDy forecast')
        ###axs[0].plot(state_truee["train_t"].numpy(), predictions[:, 0] , color="C2", label='SINDy estimate')


        for i in range(1):
            axs[0].plot(state_truee["t"].numpy(), state_truee["true_state"].numpy()[:, i], 'k--', label='true state' if i == 0 else "_nolegend_")
        for i in range(1):
            axs[0].plot(t_eval.numpy(), pred_mean_1.numpy()[:, i], "C1", label='forecast' if i == 0 else "_nolegend_")



        for j in range(1):
            axs[0].fill_between(state_truee["train_t"].numpy(), lb[:,j].numpy(), ub[:,j].numpy(), alpha=0.2, color="C0")
            axs[0].fill_between(t_eval.numpy(), pred_lb[:,j].numpy(), pred_ub[:,j].numpy(), alpha=0.2, color="C1")

        axs[0].set_xlabel("time")
        axs[0].set_ylabel("latent state")
        axs[0].legend()

        #axs[1].plot(state_truee["valid_t"].numpy(), predictions ,color="C2")

        for i in range(N):
            axs[1].plot(state_truee["train_t"].numpy(), mu.numpy()[:, i], 'C0', label='state estimate' if i == 0 else "_nolegend_")


        for i in range(N):
            axs[1].plot(state_truee["t"].numpy(), state_truee["true_state"].numpy()[:, i], 'k--', label='true state' if i == 0 else "_nolegend_")
        for i in range(N):
            axs[1].plot(t_eval.numpy(), pred_mean_1.numpy()[:, i], "C1", label='forecast' if i == 0 else "_nolegend_")



        for j in range(N):
            axs[1].fill_between(state_truee["train_t"].numpy(), lb[:,j].numpy(), ub[:,j].numpy(), alpha=0.2, color="C0")
            axs[1].fill_between(t_eval.numpy(), pred_lb[:,j].numpy(), pred_ub[:,j].numpy(), alpha=0.2, color="C1")
        ###sindy_handle = Line2D([], [], color='green', linestyle='-', label='SINDy estimate')
        handles, labels = axs[1].get_legend_handles_labels()
        ###handles.append(sindy_handle)
        ###labels.append('Sindy')
        # Aggiungi la legenda con il handle personalizzato
        axs[1].set_xlabel("Indices of Evaluated Points")
        axs[1].set_ylabel("Latent state values")
        axs[1].legend(handles, labels)
        axs[0].grid(True)
        axs[1].grid(True)
        
        file_path = f"{folder_path}/andamenti_1s_{tempo}_{num_dati}_{SNR}.png"
        plt.savefig(file_path)
        
        ##########################################################################################################################################################
        true_values = state_truee["valid_state"]
        assert true_values.shape == pred_mean_1.shape, "Le dimensioni di true_values e pred_mean devono corrispondere"

        num_elements = true_values.shape[0]

        rms_1 = np.zeros(num_elements)
        rms_sindy_1 = np.zeros(num_elements)
        uncertainty_1 = np.zeros(num_elements)
        uncertainty_sindy_1 = np.zeros(num_elements)
        #x_train = state_truee["valid_state"].detach().cpu().numpy()
        #x0 = x_train[0, :]  # Condizioni iniziali (primo punto di dati)

        #print("simulated data", simulated_data.size)
        #print("true values", true_values.size)

        for i in range(num_elements):
            residuals = pred_mean_1[i] - true_values[i]
            true_values_np = true_values.detach().cpu().numpy()
            ###residuals_sindy = simulated_data[i] - true_values_np[i]

            squared_residuals = residuals ** 2
            ###squared_residuals_sindy = residuals_sindy ** 2
            mean_squared_error = squared_residuals.mean()
            ###mean_sqared_error_sindy = squared_residuals_sindy.mean()
            ###mean_squared_error_sindy = np.float64(mean_sqared_error_sindy)  # some_value è un esempio
            ###mean_squared_error_sindy_tensor = torch.tensor(mean_squared_error_sindy, dtype=torch.float64)
            rmse_1 = torch.sqrt(mean_squared_error)
            ###rmse_sindy_1 = torch.sqrt(mean_squared_error_sindy_tensor)
            rms_1[i] = rmse_1
            ###rms_sindy_1[i] = rmse_sindy_1
            uncertainty_1[i] = squared_residuals.std() / torch.sqrt(torch.tensor(2 * squared_residuals.numel()).float())
            ###uncertainty_sindy_1[i] = squared_residuals_sindy.std() / torch.sqrt(torch.tensor(2 * squared_residuals_sindy.size).float())
        
        plt.figure(figsize=(10, 5))
        ###plt.errorbar(x=np.arange(num_elements), y=rms_sindy_1, yerr=uncertainty_sindy_1, fmt='-o', label='RMS SINDy', color='red', ecolor='red', capsize=5)
        plt.errorbar(x=np.arange(num_elements), y=rms_1, yerr=uncertainty_1, fmt='-o', label='RMS SVISE', color='blue', ecolor='blue', capsize=5)
        plt.xlabel('Indice Temporale')
        plt.ylabel('RMS')
        plt.title('RMS')
        plt.legend()
        plt.grid()
        #plt.show()

        # Definisci il nome del file includendo parametri dinamici
        file_path = f"{folder_path}/RMS_1s_{tempo}_{num_dati}_{SNR}.png"
        plt.savefig(file_path, bbox_inches='tight')
        rms_means_1.append(rms_1)
        sigma_means_1.append(uncertainty_1)
        ###rms_means_sindy_1.append(rms_sindy_1)
        ###sigma_means_sindy_1.append(uncertainty_sindy_1)


        file_nome = f"{folder_path}/Y_1s_{tempo}_{SNR}.txt"
        #np.save(f"{folder_path}/.npy", total_sum_final_values_100_valori)

        ###np.savetxt(file_nome, np.column_stack((rms_1, uncertainty_1, rms_sindy_1, uncertainty_sindy_1)), comments='', fmt='%f')
        
        #####################################################################################################################################################

        RMS = []
        sigmaRMS = []
        state_true = data #dizionario(num_dati, tempo, m)
        x_train = state_true["train_y"].detach().cpu().numpy()
        dt = state_true["train_t"].detach().cpu().numpy()
        x0 = x_train[0, :]  # Condizioni iniziali (primo punto di dati)
        #scaler = StandardScaler()
        #x_train_scaled = scaler.fit_transform(x_train)
        optimizer = ps.STLSQ(threshold=0.1, alpha=0.2, max_iter=100000)
        ###model = ps.SINDy(feature_library=combined_library, optimizer=optimizer)
        
        ###model.fit(x_train, t=dt)
        ###model.print()

        ###predictions = model.predict(x_train)
        ###predictions = model.differentiate(x_train, t=dt)
        #x_train = state_true["valid_state"].detach().cpu().numpy()
        x0 = x_train[0, :]  # Condizioni iniziali (primo punto di dati)
        #print("true t",state_true["train_t"][0])
        #print("truee t",state_truee["train_t"][0])

        #predictions = model.simulate(x0, dt)
        x0 = state_true["valid_state"].detach().cpu().numpy()
        y0 = x0[0,:] # y0 era da predictions-1 
        #y0 = state_true["train_y"]
        #print(x0)
        dtt = state_true["valid_t"].detach().cpu().numpy()
        ###simulated_data = model.simulate(y0, dtt)

        K=32  #num_dati
        with torch.no_grad():
            # get the state estimate
            mu = sde.marginal_sde.mean(state_true["train_t"])
            var = sde.marginal_sde.K(state_true["train_t"]).diagonal(dim1=-2, dim2=-1)
            lb = mu - 2 * var.sqrt()
            ub = mu + 2 * var.sqrt()

            # generate a forecast using 32 samples
            x0 = mu[-1] + torch.randn(num_mc_samples, N) * var[-1].sqrt() ################# 30????????????????????????????''
            sde_kwargs = dict(dt=1e-2, atol=1e-2, rtol=1e-2, adaptive=True)
            t_eval = torch.linspace(state_true["train_t"].max(), state_true["t"].max(), len(state_true["valid_state"]))  #38.4 x 4
            xs = sdeint.solve_sde(sde, x0, t_eval, **sde_kwargs)  #sdeint.solve_sde
            pred_mean = xs.mean(1)
            pred_lb = pred_mean - 2 * xs.std(1) 
            pred_ub = pred_mean + 2 * xs.std(1) 

        #########################################################################################################
        
        true_values = state_true["valid_state"]
        assert true_values.shape == pred_mean.shape, "Le dimensioni di true_values e pred_mean devono corrispondere"

        num_elements = true_values.shape[0]

        rms = np.zeros(num_elements)
        ###rms_sindy = np.zeros(num_elements)
        uncertainty = np.zeros(num_elements)
        ###uncertainty_sindy = np.zeros(num_elements)
        #x_train = state_true["valid_state"].detach().cpu().numpy()
        #x0 = x_train[0, :]  # Condizioni iniziali (primo punto di dati)

        for i in range(num_elements):
            residuals = pred_mean[i] - true_values[i]
            true_values_np = true_values.detach().cpu().numpy()
            ###residuals_sindy = simulated_data[i] - true_values_np[i]
            squared_residuals = residuals ** 2
            ###squared_residuals_sindy = residuals_sindy ** 2
            mean_squared_error = squared_residuals.mean()
            ###mean_sqared_error_sindy = squared_residuals_sindy.mean()
            ###mean_squared_error_sindy = np.float64(mean_sqared_error_sindy)  # some_value è un esempio
            ###mean_squared_error_sindy_tensor = torch.tensor(mean_squared_error_sindy, dtype=torch.float64)
            rmse = torch.sqrt(mean_squared_error)
            ###rmse_sindy = torch.sqrt(mean_squared_error_sindy_tensor)
            rms[i] = rmse
            ###rms_sindy[i] = rmse_sindy
            uncertainty[i] = squared_residuals.std() / torch.sqrt(torch.tensor(2 * squared_residuals.numel()).float())
            ###uncertainty_sindy[i] = squared_residuals_sindy.std() / torch.sqrt(torch.tensor(2 * squared_residuals_sindy.size).float())

        x=np.arange(num_elements)  ###############################
        plt.figure(figsize=(10, 5))
        ###plt.errorbar(x=x, y=rms_sindy, yerr=uncertainty_sindy, fmt='-o', label='RMS SINDy', color='red', ecolor='red', capsize=5)
        plt.errorbar(x=x, y=rms, yerr=uncertainty, fmt='-o', label='RMS SVISE', color='blue', ecolor='blue', capsize=5)
        plt.xlabel('Indice Temporale')
        plt.ylabel('RMS')
        plt.title('RMS')
        plt.legend()
        plt.grid()
        file_path = f"{folder_path}/RMS_{tempo}_{num_dati}_{SNR}.png"
        plt.savefig(file_path, bbox_inches='tight')
        #plt.show()



        file_nome = f"{folder_path}/Y_{tempo}_{SNR}.txt"

        ###np.savetxt(file_nome, np.column_stack((rms, uncertainty, rms_sindy, uncertainty_sindy)), comments='', fmt='%f')
        
        #########################################################################################################

        rms_means.append(rms)
        sigma_means.append(uncertainty)
        ###rms_means_sindy.append(rms_sindy)
        ###sigma_means_sindy.append(uncertainty_sindy)


        # Assumi che queste variabili siano già definite
        num_nodi = N  # Assicurati che N sia definito correttamente come il numero di nodi
        sde.eval()
        

        color_segments = [inferno(x) for x in np.linspace(0, 0.7, 4)]


        total_sum_final_values = []
        total_sum_finale_values = []
        total_sum_final_values_sindy = []
        total_sum_finale_values_sindy = []
        
        def rebuild_adjacency_matrix_sindy_coeff(coefficients):
            adjacency_matrix = np.zeros((N, N))
            for i in range(N):
                row = coefficients[i, :-1]  # Escludi l'ultimo termine che è costante
                for j, coeff in enumerate(row):
                    if coeff != 0:
                        # Estrai gli indici delle variabili utilizzando regex
                        matches = re.findall(r'\d+', combined_sin_labels[j])
                        if len(matches) == 2:  # Assicurati di avere esattamente due indici
                            index_i = int(matches[0]) - 1
                            index_j = int(matches[1]) - 1
                            
                            # Scegli l'indice corretto basato sul segno del coefficiente
                            if coeff < -threshold:
                                # Somma il valore negativo del coefficiente
                                adjacency_matrix[index_i, index_j] += -coeff
                            if coeff > threshold:
                                # Somma il valore del coefficiente
                                adjacency_matrix[index_j, index_i] += coeff
            sindy_coeff = adjacency_matrix
            return sindy_coeff
        
        def rebuild_adjacency_matrix_sindy(coefficients):
            adjacency_matrix = np.zeros((N, N))
            for i in range(N):
                row = coefficients[i, :-1]  # Escludi l'ultimo termine che è costante
                for j, coeff in enumerate(row):
                    if coeff != 0:
                        # Estrai gli indici delle variabili utilizzando regex
                        matches = re.findall(r'\d+', combined_sin_labels[j])
                        if len(matches) == 2:  # Assicurati di avere esattamente due indici
                            index_i = int(matches[0]) - 1
                            index_j = int(matches[1]) - 1
                            
                            # Scegli l'indice corretto basato sul segno del coefficiente
                            if coeff < -threshold:
                                # Somma il valore negativo del coefficiente
                                adjacency_matrix[index_i, index_j] = 1
                            if coeff > threshold:
                                # Somma il valore del coefficiente
                                adjacency_matrix[index_j, index_i] = 1
            sindy_coeff = adjacency_matrix
            return sindy_coeff
        
        def rebuild_adjacency_matrix(calculated_terms, size):
            new_A = np.zeros((size, size))
            for term, coeff in calculated_terms.items():
                # Assumi che il termine abbia la forma 'sin(theta_i - theta_j)'
                match = re.search(r'theta_(\d+) - theta_(\d+)', term)
                if match:
                    i, j = map(int, match.groups())
                    i, j = i - 1, j - 1  # Adjust index to zero-based
                    if abs(coeff) >= threshold:
                        '''
                        if new_A[i, j] != 0:
                            new_A[j, i] = 1
                        else:
                            new_A[i, j] = 1 #coeff
                            '''
                        if (coeff) >= 0:
                            new_A[i, j] = 1
                        else:
                            new_A[j, i] = 1

            return new_A
        
        def rebuild_adjacency_matrix_coeff(calculated_terms, size):
            new_A = np.zeros((size, size))
            #accum_matrix = np.zeros((size, size))

            for term, coeff in calculated_terms.items():
                # Assumi che il termine abbia la forma 'sin(theta_i - theta_j)'
                match = re.search(r'theta_(\d+) - theta_(\d+)', term)
                if match:
                    i, j = map(int, match.groups())
                    i, j = i - 1, j - 1  # Adjust index to zero-based
                    if abs(coeff) >= threshold:
                        if (coeff) >= 0:
                            new_A[i, j] += abs(coeff)
                        else:
                            new_A[j, i] += abs(coeff)

                        #new_A[j, i] = coeff  # Ensure symmetry
                        #print(f"Term {term} with coeff {coeff} added at positions ({i},{j}) and ({j},{i})")
            #print("new A",accum_matrix)
            return new_A


        def parse_terms(term_string, threshold):
            # Extract terms and coefficients from the equation string
            terms = re.findall(r"([+-]?[\d\.]+)sin\(theta_(\d+) - theta_(\d+)\)", term_string)
            # Filter and store terms with coefficients above the threshold
            parsed_terms = {f"sin(theta_{i} - theta_{j})": float(coeff) 
                            for coeff, i, j in terms if abs(float(coeff)) >= threshold}
            #print(f"Terms extracted from '{term_string}': {parsed_terms}")  # Stampa per debug
            return parsed_terms


        def generate_correct_terms_strings(A):
            # Generate a list of sets of correct terms for each node based on the adjacency matrix A
            correct_terms_strings = []
            for i in range(A.shape[0]):
                terms = set()
                for j in range(A.shape[1]):
                    if A[i, j] == 1 and i < j:  # Check only one connection per pair (i < j)
                        terms.add(f"sin(theta_{i+1} - theta_{j+1})")
                    elif A[i, j] == 1 and i > j:  # Add the inverse term if not considered in i < j
                        terms.add(f"sin(theta_{j+1} - theta_{i+1})")
                correct_terms_strings.append(terms)
            #print("correct terms for node:", 0 ,correct_terms_strings[0])
            return correct_terms_strings

        def calculate_accuracy(calculated_terms, correct_terms):
            # Calculate accuracy by comparing calculated terms with correct terms
            calculated_terms_set = set(calculated_terms.keys())  # Dictionary keys of calculated terms
            correct_terms_set = correct_terms  # Already a set
            matched_terms = calculated_terms_set.intersection(correct_terms_set)
            total_correct = len(correct_terms_set)
            matched_count = len(matched_terms)
            #print("matched terms", (len(matched_terms)))
            #print("calculated", len(calculated_terms))
            return matched_count / len(calculated_terms) if len(calculated_terms) > 0 else 0


        def sum_reconstructed_matrices(reconstructed_matrices):
            if len(reconstructed_matrices) == 0:
                return None
            abs_matrices = [np.abs(matrix) for matrix in reconstructed_matrices]
            summed_matrix = np.sum(abs_matrices, axis=0)  #######diff?
            return summed_matrix


        def veri_positivi(calculated_terms, correct_terms):
            calculated_terms_set = set(calculated_terms.keys())  # Dictionary keys of calculated terms
            correct_terms_set = correct_terms  # Already a set
            matched_terms = calculated_terms_set.intersection(correct_terms_set)   
            return len(matched_terms)

        def falsi_positivi(calculated_terms,correct_terms):
            calculated_terms_set = set(calculated_terms.keys())  # Dictionary keys of calculated terms
            correct_terms_set = correct_terms  # Already a set
            falsi_positivi = calculated_terms_set.difference(correct_terms_set)
            return len(falsi_positivi)

        def falsi_negativi(calculated_terms,correct_terms):
            calculated_terms_set = set(calculated_terms.keys())  # Dictionary keys of calculated terms
            correct_terms_set = correct_terms  # Already a set
            falsi_negativi = correct_terms_set.difference(calculated_terms_set)
            return len(falsi_negativi)

        def calculate_accuratezza(calculated_terms, correct_terms):
            # Calculate accuracy by comparing calculated terms with correct terms
            calculated_terms_set = set(calculated_terms.keys())  # Dictionary keys of calculated terms
            correct_terms_set = correct_terms  # Already a set
            matched_terms = calculated_terms_set.intersection(correct_terms_set)
            total_correct = len(correct_terms_set)
            matched_count = len(matched_terms)
            #print("matched terms", (len(matched_terms)))
            #print("calculated", len(calculated_terms))
            return matched_count / len(correct_terms) if len(correct_terms) > 0 else 0
        ##########################################       ricostruzione matrice sindy        ###############################################
        '''
        coefficients = model.coefficients()
        feature_names = model.get_feature_names()
        # Assumiamo che ci sia N oscillatori
        N = x_train.shape[1]
        adjacency_matrix_sindy_coeff = np.zeros((N, N))
        t_values = np.arange(0, 6.01, 0.01)
        max_coefficient = np.max(np.abs(coefficients))
        t_valori = np.arange(0, max_coefficient, 0.01)
        
        #combined_t = np.concatenate((t_values, t_valori))
        # Assegnazione dei coefficienti alla matrice di adiacenza con somma
        for i in range(N):
            row = coefficients[i, :-1]  # Escludi l'ultimo termine che è costante
            for j, coeff in enumerate(row):
                if coeff != 0:
                    # Estrai gli indici delle variabili utilizzando regex
                    matches = re.findall(r'\d+', combined_sin_labels[j])
                    if len(matches) == 2:  # Assicurati di avere esattamente due indici
                        index_i = int(matches[0]) - 1
                        index_j = int(matches[1]) - 1
                        
                        # Scegli l'indice corretto basato sul segno del coefficiente
                        if coeff < 0:
                            # Somma il valore negativo del coefficiente
                            adjacency_matrix_sindy_coeff[index_i, index_j] += -coeff
                        else:
                            # Somma il valore del coefficiente
                            adjacency_matrix_sindy_coeff[index_j, index_i] += coeff
        sindy_coeff = adjacency_matrix_sindy_coeff
        #print("Matrice di adiacenza ricostruita e aggiornata con coefficienti:")
        #print(sindy_coeff)
        #np.save('path_to_save/sindy_coeff_matrix.npy', sindy_coeff)
        np.save(f"{folder_path}/sindy_coeff_matrix_{num_dati}_{SNR}.npy", sindy_coeff)




        adjacency_matrix_sindy = np.zeros((N, N))

        # Assegnazione dei coefficienti alla matrice di adiacenza con somma
        for i in range(N):
            row = coefficients[i, :-1]  # Escludi l'ultimo termine che è costante
            for j, coeff in enumerate(row):
                if coeff != 0:
                    # Estrai gli indici delle variabili utilizzando regex
                    matches = re.findall(r'\d+', combined_sin_labels[j])
                    if len(matches) == 2:  # Assicurati di avere esattamente due indici
                        index_i = int(matches[0]) - 1
                        index_j = int(matches[1]) - 1
                        
                        # Scegli l'indice corretto basato sul segno del coefficiente
                        if coeff < 0:
                            # Somma il valore negativo del coefficiente
                            adjacency_matrix_sindy[index_i, index_j] = 1
                        else:
                            # Somma il valore del coefficiente
                            adjacency_matrix_sindy[index_j, index_i] = 1
        sindy = adjacency_matrix_sindy
        np.save(f"{folder_path}/sindy_matrix_{num_dati}_{SNR}.npy", sindy)

        #print("Matrice di adiacenza ricostruita:")
        #print(sindy)
        
        ####################################################################################################################################
        plt.figure(figsize=(3, 3))  # Imposta la dimensione del grafico
        plt.imshow(sindy, cmap='grey')  # Usa la colormap 'inferno'
        plt.colorbar()  # Aggiungi una barra dei colori per mostrare la scala dei valori
        plt.title('Matrice SINDy')
        plt.xlabel('Indice ')
        file_path = f"{folder_path}/matrice SINDy_{tempo}_{SNR}_{num_dati}_{SNR}.png"
        plt.savefig(file_path)
        plt.close()

        plt.figure(figsize=(3, 3))  # Imposta la dimensione del grafico
        plt.imshow(sindy_coeff, cmap='inferno')  # Usa la colormap 'inferno'
        plt.colorbar()  # Aggiungi una barra dei colori per mostrare la scala dei valori
        plt.title('Matrice SINDy')
        plt.xlabel('Indice ')
        file_path = f"{folder_path}/matrice SINDy_coeff_{tempo}_{SNR}_{num_dati}_{SNR}.png"

        # Salva il plot nel file
        plt.savefig(file_path)
        plt.close()
        '''
        accuracies = []
        accuratezze = []
        Forbenius_metric = []
        Forbenius_metric_sindy = []
        Forbenius_coeff = []
        Forbenius_coeff_sindy = []
        f1 = []
        reconstructed_matrices = []
        reconstructed_matrices_coeff = []
        correct_terms_strings = generate_correct_terms_strings(A)
        #print(sde_prior_feature_names)
        t_values = np.arange(0, 6.01, 0.01)

        max_valori = np.max(t_values)

        for t in t_values:
            threshold = t 
            Forbenius_metric = []
            Forbenius_coeff = []
            ###sindy = rebuild_adjacency_matrix_sindy(coefficients)
            ###sindy_coeff = rebuild_adjacency_matrix_sindy_coeff(coefficients)

            for node_index, term_string in enumerate(sde_prior_feature_names):
                calculated_terms = parse_terms(term_string, threshold)
                reconstructed_A = rebuild_adjacency_matrix(calculated_terms, A.shape[0])
                reconstructed_A_coeff = rebuild_adjacency_matrix_coeff(calculated_terms, A.shape[0])
                reconstructed_matrices.append(reconstructed_A)
                reconstructed_matrices_coeff.append(reconstructed_A_coeff)
                final_matrix = np.sum(reconstructed_matrices_coeff, axis=0)
                #print("Somma finale delle matrici:", final_matrix)

                # Usa node_index-offset per accedere ai termini corretti
                correct_index = node_index 
                accuracy = calculate_accuracy(calculated_terms, correct_terms_strings[correct_index])
                accuratezza = calculate_accuratezza(calculated_terms, correct_terms_strings[correct_index])
                accuracies.append(accuracy)
                accuratezze.append(accuratezza)

                if (veri_positivi(calculated_terms, correct_terms_strings[correct_index]) + falsi_positivi(calculated_terms, correct_terms_strings[correct_index])) != 0:
                    precision = veri_positivi(calculated_terms, correct_terms_strings[correct_index]) / (veri_positivi(calculated_terms, correct_terms_strings[correct_index]) + falsi_positivi(calculated_terms, correct_terms_strings[correct_index]))
                else:
                    precision = 0

                if (veri_positivi(calculated_terms, correct_terms_strings[correct_index]) + falsi_negativi(calculated_terms, correct_terms_strings[correct_index])) != 0:
                    recall = veri_positivi(calculated_terms, correct_terms_strings[correct_index]) / (veri_positivi(calculated_terms, correct_terms_strings[correct_index]) + falsi_negativi(calculated_terms, correct_terms_strings[correct_index]))
                else:
                    recall = 0

                if precision + recall != 0:
                    F1 = 2 * precision * recall / (precision + recall)
                    f1.append(F1)
                else:
                    F1 = 0
                    f1.append(F1)
                


        
        # Calculate the average accuracy
            mean_accuracy = np.mean(accuracies)
            mean_accuratezza = np.mean(accuratezze)
            #print("Average Accuracy:", mean_accuracy)
            #print("Average Accuracy with respect to the right terms:",mean_accuratezza)
            mean_F1 = np.mean(f1)
            #print("Average F1: ", mean_F1*100)
        #print(reconstructed_matrices)
            
            final_reconstructed_matrix = sum_reconstructed_matrices(reconstructed_matrices)
            final_reconstructed_matrix_coeff = sum_reconstructed_matrices(reconstructed_matrices_coeff)

            finale = np.abs((A)-np.abs(reconstructed_A))
            finale_coeff = np.abs((A)-np.abs(reconstructed_A_coeff))
            ###finale_sindy = np.abs((A)-np.abs(sindy))
            #print("finale sindy",finale_sindy)
            ###finale_coeff_sindy = np.abs((A)-np.abs(sindy_coeff))

            abs_matrices_coeff = [(matrix) for matrix in reconstructed_matrices_coeff]
            final_matrix_coeff = np.sum(abs_matrices_coeff, axis=0)

            abs_matrices = [(matrix) for matrix in reconstructed_matrices]
            final_matrix = np.sum(abs_matrices, axis=0)

            total_sum_A = 0.0
            N = A.shape[0]  # Assicurati che questa definizione sia corretta e consistente in tutto il codice

            for i in range(N):
                for j in range(N):
                    total_sum_A += A[i, j] ** 2
            #print("Somma dei quadrati degli elementi di A:", total_sum_A)

        # Calcolo della somma dei quadrati degli elementi di final_reconstructed_matrix
            total_sum_final = 0.0
            for i in range(N):
                for j in range(N):
                    total_sum_final += finale[i, j] ** 2
            #print("Somma dei quadrati degli elementi di final_reconstructed_matrix:", total_sum_final)

        # Calcolo della somma dei quadrati degli elementi della matrice delle differenze
            total_sum_finale = 0.0
            for i in range(N):
                for j in range(N):
                    total_sum_finale += finale_coeff[i, j] ** 2

            t###otal_sum_final_sindy = 0.0
            ###for i in range(N):
            ###    for j in range(N):
            ###        #if abs(finale_sindy[i, j]) > t:
            ###            total_sum_final_sindy += finale_sindy[i, j] ** 2
            #print("Somma dei quadrati degli elementi di finale_sindy (filtrati):", total_sum_final_sindy)

            # Calcolo della somma dei quadrati degli elementi della matrice delle differenze finale_coeff_sindy, considerando solo i valori > t

            ###total_sum_finale_sindy = 0.0
            ###for i in range(N):
            ###    for j in range(N):
            ###        #if abs(finale_coeff_sindy[i, j]) > t:
            ###            total_sum_finale_sindy += finale_coeff_sindy[i, j] ** 2
            #print("Somma dei quadrati degli elementi di finale_coeff_sindy (filtrati):", total_sum_finale_sindy)

            #print("Somma dei quadrati degli elementi della matrice delle differenze:", total_sum_finale)
            Forbenius_metric = total_sum_final/total_sum_A
            Forbenius_coeff = total_sum_finale/total_sum_A

            ###Forbenius_metric_sindy = total_sum_final_sindy/total_sum_A
            ###Forbenius_coeff_sindy = total_sum_finale_sindy/total_sum_A

            total_sum_final_values.append(Forbenius_metric)
            total_sum_finale_values.append(Forbenius_coeff)

            ###total_sum_final_values_sindy.append(Forbenius_metric_sindy)
            ###total_sum_finale_values_sindy.append(Forbenius_coeff_sindy)

            rec = torch.tensor(finale, dtype=torch.float32)
            rece = torch.tensor(finale_coeff, dtype=torch.float32)
            reconstructed = torch.tensor(reconstructed_A, dtype=torch.float32)
            reconstructed_coeff = torch.tensor(reconstructed_A_coeff, dtype=torch.float32)

            if t==0:
                plt.figure(figsize=(3, 3))  # Imposta la dimensione del grafico
                plt.imshow(final_reconstructed_matrix, cmap='inferno')  # Usa la colormap 'inferno'
                plt.colorbar()  # Aggiungi una barra dei colori per mostrare la scala dei valori
                plt.title('Matrice SVISE')
                plt.xlabel('Indice ')
                file_path = f"{folder_path}/matrice SVISE_{tempo}_{SNR}_{num_dati}_{SNR}.png"
                plt.savefig(file_path)
                plt.close()
                np.save(f"{folder_path}/svise_matrix_{num_dati}_{SNR}.npy", final_reconstructed_matrix)


                plt.figure(figsize=(3, 3))  # Imposta la dimensione del grafico
                plt.imshow(final_reconstructed_matrix_coeff, cmap='inferno')  # Usa la colormap 'inferno'
                plt.colorbar()  # Aggiungi una barra dei colori per mostrare la scala dei valori
                plt.title('Matrice SVISE coeff')
                plt.xlabel('Indice')
                file_path = f"{folder_path}/matrice SVISE_coeff_{tempo}_{SNR}_{num_dati}_{SNR}.png"
                plt.savefig(file_path)
                plt.close()
                np.save(f"{folder_path}/svise_coeff_matrix_{num_dati}_{SNR}.npy", final_reconstructed_matrix_coeff)


                #plt.show()
        #print("Somma dei quadrati degli elementi della matrice delle differenze:", Forbenius_metric)
        
            if SNR == 10:
                total_sum_final_values_10.append(Forbenius_metric)
                total_sum_finale_values_10.append(Forbenius_coeff)
                ###total_sum_final_values_10_sindy.append(Forbenius_metric_sindy)
                ###total_sum_finale_values_10_sindy.append(Forbenius_coeff_sindy)
                if t == max_valori:
                    np.save(f"{folder_path}/total_sum_final_values_10.npy", total_sum_final_values_10)
                    np.save(f"{folder_path}/total_sum_finale_values_10.npy", total_sum_finale_values_10)
                    ###np.save(f"{folder_path}/total_sum_final_values_10_sindy.npy", total_sum_final_values_10_sindy)
                    ###np.save(f"{folder_path}/total_sum_finale_values_10_sindy.npy", total_sum_finale_values_10_sindy) 
            if SNR ==20:
                total_sum_final_values_20.append(Forbenius_metric)
                total_sum_finale_values_20.append(Forbenius_coeff)
                ###total_sum_final_values_20_sindy.append(Forbenius_metric_sindy)
                ###total_sum_finale_values_20_sindy.append(Forbenius_coeff_sindy) 
                if t == max_valori:
                    np.save(f"{folder_path}/total_sum_final_values_20.npy", total_sum_final_values_20)
                    np.save(f"{folder_path}/total_sum_finale_values_20.npy", total_sum_finale_values_20)
                    ###np.save(f"{folder_path}/total_sum_final_values_20_sindy.npy", total_sum_final_values_20_sindy)
                    ###np.save(f"{folder_path}/total_sum_finale_values_20_sindy.npy", total_sum_finale_values_20_sindy) 
            if SNR == 100:
                total_sum_final_values_100.append(Forbenius_metric)
                total_sum_finale_values_100.append(Forbenius_coeff)
                ###total_sum_final_values_100_sindy.append(Forbenius_metric_sindy)
                ###total_sum_finale_values_100_sindy.append(Forbenius_coeff_sindy) 
                if t == max_valori:
                    np.save(f"{folder_path}/total_sum_final_values_100.npy", total_sum_final_values_100)
                    np.save(f"{folder_path}/total_sum_finale_values_100.npy", total_sum_finale_values_100)
                    ###np.save(f"{folder_path}/total_sum_final_values_100_sindy.npy", total_sum_final_values_100_sindy)
                    ###np.save(f"{folder_path}/total_sum_finale_values_100_sindy.npy", total_sum_finale_values_100_sindy) 
        del sde  # Elimina l'oggetto sde
        
        import gc  # Importa il modulo del garbage collector
        gc.collect()  # Chiama il garbage collector per liberare la memoria
        colors = inferno(np.linspace(0.2, 0.8, 2)) 
        # Plotting dei risultati
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(t_values, total_sum_final_values, label='Frobenius metric SVISE',color=colors[0], linewidth=2)
        ###plt.plot(t_values, total_sum_final_values_sindy, label='Frobenius metric SINDy',color=colors[1], linewidth=2)
        plt.title('Frobenius-metric')
        plt.xlabel('t')
        plt.ylabel('Frobenius-metric')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t_values, total_sum_finale_values, label='Frobenius coeff SVISE', color=colors[0], linewidth=2)
        ###plt.plot(t_values, total_sum_finale_values_sindy, label='Frobenius coeff SINDy', color=colors[1], linewidth=2)
        plt.title('Frobenius-metric-coeff vs t')
        plt.xlabel('t')
        plt.ylabel('Frobenius-metric-coeff')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
        plt.legend()

        plt.tight_layout()

        # Definisci il nome del file includendo parametri dinamici
        file_path = f"{folder_path}/metriche_{num_dati}_{SNR}.png"

        # Codice per generare il plot va qui
        # plt.plot(...) o altre funzioni di plotting

        # Salva il plot nel file
        plt.savefig(file_path)
        #plt.show()
#################################################################

        ###max_coefficient = np.max(np.abs(coefficients))
        t_valori = np.arange(0, 16, 0.1)
        max_valori = np.max(t_valori)
        for t in t_valori:
            threshold = t 
            Forbenius_metric = []
            Forbenius_coeff = []
            ###sindy = rebuild_adjacency_matrix_sindy(coefficients)
            ###sindy_coeff = rebuild_adjacency_matrix_sindy_coeff(coefficients)

            for node_index, term_string in enumerate(sde_prior_feature_names):
                calculated_terms = parse_terms(term_string, threshold)
                reconstructed_A = rebuild_adjacency_matrix(calculated_terms, A.shape[0])
                reconstructed_A_coeff = rebuild_adjacency_matrix_coeff(calculated_terms, A.shape[0])
                reconstructed_matrices.append(reconstructed_A)
                reconstructed_matrices_coeff.append(reconstructed_A_coeff)
                final_matrix = np.sum(reconstructed_matrices_coeff, axis=0)




            
            final_reconstructed_matrix = sum_reconstructed_matrices(reconstructed_matrices)
            final_reconstructed_matrix_coeff = sum_reconstructed_matrices(reconstructed_matrices_coeff)

            finale = np.abs((A)-np.abs(reconstructed_A))
            finale_coeff = np.abs((A)-np.abs(reconstructed_A_coeff))
            ###finale_sindy = np.abs((A)-np.abs(sindy))
            #print("finale sindy",finale_sindy)
            ###finale_coeff_sindy = np.abs((A)-np.abs(sindy_coeff))
            #print("A?",A , " sindycoeff ", sindy_coeff)
            #print("finale coeff??", finale_coeff_sindy)
            abs_matrices_coeff = [(matrix) for matrix in reconstructed_matrices_coeff]
            final_matrix_coeff = np.sum(abs_matrices_coeff, axis=0)

            abs_matrices = [(matrix) for matrix in reconstructed_matrices]
            final_matrix = np.sum(abs_matrices, axis=0)

            total_sum_A = 0.0
            N = A.shape[0]  # Assicurati che questa definizione sia corretta e consistente in tutto il codice

            for i in range(N):
                for j in range(N):
                    total_sum_A += A[i, j] ** 2
            #print("Somma dei quadrati degli elementi di A:", total_sum_A)

        # Calcolo della somma dei quadrati degli elementi di final_reconstructed_matrix
            total_sum_final = 0.0
            for i in range(N):
                for j in range(N):
                    total_sum_final += finale[i, j] ** 2
            #print("Somma dei quadrati degli elementi di final_reconstructed_matrix:", total_sum_final)

        # Calcolo della somma dei quadrati degli elementi della matrice delle differenze
            total_sum_finale = 0.0
            for i in range(N):
                for j in range(N):
                    total_sum_finale += finale_coeff[i, j] ** 2

            ###total_sum_final_sindy = 0.0
            ###for i in range(N):
            ###    for j in range(N):
            ###        #if abs(finale_sindy[i, j]) > t:
            ###            total_sum_final_sindy += finale_sindy[i, j] ** 2
            #print("Somma dei quadrati degli elementi di finale_sindy (filtrati):", total_sum_final_sindy)

            # Calcolo della somma dei quadrati degli elementi della matrice delle differenze finale_coeff_sindy, considerando solo i valori > t
            total_sum_finale_sindy = 0.0
            ###for i in range(N):
            ###    for j in range(N):
            ###        #if abs(finale_coeff_sindy[i, j]) > t:
            ###            total_sum_finale_sindy += finale_coeff_sindy[i, j] ** 2
            #print("Somma dei quadrati degli elementi di finale_coeff_sindy (filtrati):", total_sum_finale_sindy)

            #print("Somma dei quadrati degli elementi della matrice delle differenze:", total_sum_finale)
            Forbenius_metric = total_sum_final/total_sum_A
            Forbenius_coeff = total_sum_finale/total_sum_A

            ###Forbenius_metric_sindy = total_sum_final_sindy/total_sum_A
            ###Forbenius_coeff_sindy = total_sum_finale_sindy/total_sum_A

            total_sum_final_values_valori.append(Forbenius_metric)
            total_sum_finale_values_valori.append(Forbenius_coeff)

            ###total_sum_final_values_sindy_valori.append(Forbenius_metric_sindy)
            ###total_sum_finale_values_sindy_valori.append(Forbenius_coeff_sindy)

            '''
            rec = torch.tensor(finale, dtype=torch.float32)
            rece = torch.tensor(finale_coeff, dtype=torch.float32)
            reconstructed = torch.tensor(reconstructed_A, dtype=torch.float32)
            reconstructed_coeff = torch.tensor(reconstructed_A_coeff, dtype=torch.float32)
            '''
            if t == 0:
                '''
                plt.figure(figsize=(3, 3))  # Imposta la dimensione del grafico
                plt.imshow(final_reconstructed_matrix, cmap='inferno')  # Usa la colormap 'inferno'
                plt.colorbar()  # Aggiungi una barra dei colori per mostrare la scala dei valori
                plt.title('Matrice SVISE')
                plt.xlabel('Indice Colonna')
                plt.ylabel('Indice Riga')
                file_path = f"{folder_path}/matrice SVISE_{tempo}_{SNR}_{num_dati}_{SNR}.png"
                plt.savefig(file_path)
                plt.close()

                plt.figure(figsize=(3, 3))  # Imposta la dimensione del grafico
                plt.imshow(final_reconstructed_matrix_coeff, cmap='inferno')  # Usa la colormap 'inferno'
                plt.colorbar()  # Aggiungi una barra dei colori per mostrare la scala dei valori
                plt.title('Matrice SVISE coeff')
                plt.xlabel('Indice Colonna')
                plt.ylabel('Indice Riga')
                file_path = f"{folder_path}/matrice SVISE_coeff_{tempo}_{SNR}_{num_dati}_{SNR}.png"
                plt.savefig(file_path)
                plt.close()
                '''
        
            if SNR == 10:
                #print("debug frobenius",Forbenius_metric)
                total_sum_final_values_10_valori.append(Forbenius_metric)
                total_sum_finale_values_10_valori.append(Forbenius_coeff)           
                ###total_sum_final_values_10_sindy_valori.append(Forbenius_metric_sindy)
                ###total_sum_finale_values_10_sindy_valori.append(Forbenius_coeff_sindy)   
                if t == max_valori:
                    np.save(f"{folder_path}/total_sum_final_values_10_valori.npy", total_sum_final_values_10_valori)
                    np.save(f"{folder_path}/total_sum_finale_values_10_valori.npy", total_sum_finale_values_10_valori)
                    ###np.save(f"{folder_path}/total_sum_final_values_10_sindy_valori.npy", total_sum_final_values_10_sindy_valori)
                    ###np.save(f"{folder_path}/total_sum_finale_values_10_sindy_valori.npy", total_sum_finale_values_10_sindy_valori)
                    #print("valori 10", total_sum_final_values_10_sindy_valori)

            if SNR ==20:
                total_sum_final_values_20_valori.append(Forbenius_metric)
                total_sum_finale_values_20_valori.append(Forbenius_coeff)
                ###total_sum_final_values_20_sindy_valori.append(Forbenius_metric_sindy)
                ###total_sum_finale_values_20_sindy_valori.append(Forbenius_coeff_sindy) 
                if t == max_valori:
                    np.save(f"{folder_path}/total_sum_final_values_20_valori.npy", total_sum_final_values_20_valori)
                    np.save(f"{folder_path}/total_sum_finale_values_20_valori.npy", total_sum_finale_values_20_valori)
                    ###np.save(f"{folder_path}/total_sum_final_values_20_sindy_valori.npy", total_sum_final_values_20_sindy_valori)
                    ###np.save(f"{folder_path}/total_sum_finale_values_20_sindy_valori.npy", total_sum_finale_values_20_sindy_valori)
            if SNR == 100:
                total_sum_final_values_100_valori.append(Forbenius_metric)
                total_sum_finale_values_100_valori.append(Forbenius_coeff)
                ###total_sum_final_values_100_sindy_valori.append(Forbenius_metric_sindy)
                ###total_sum_finale_values_100_sindy_valori.append(Forbenius_coeff_sindy) 
                if t == max_valori:
                    np.save(f"{folder_path}/total_sum_final_values_100_valori.npy", total_sum_final_values_100_valori)
                    np.save(f"{folder_path}/total_sum_finale_values_100_valori.npy", total_sum_finale_values_100_valori)
                    ###np.save(f"{folder_path}/total_sum_final_values_100_sindy_valori.npy", total_sum_final_values_100_sindy_valori)
                    ###np.save(f"{folder_path}/total_sum_finale_values_100_sindy_valori.npy", total_sum_finale_values_100_sindy_valori)


    #plt.show()
    colors = inferno(np.array([0, 0.3, 0.7]))

        # Configurazione della figura
    plt.figure(figsize=(10, 6))
    gruppo = ["SNR 10", "SNR 20", "SNR 100"]
    full_time = state_true["valid_t"]  # Array completo dei tempi
    num_points = [len(rms_means[i]) for i in range(3)]

    for i in range(3):  # Itera per i 3 gruppi
        interval = len(full_time) / num_points[i]
        indices = np.round(np.arange(0, len(full_time), interval)).astype(int)
        valid_t_segment = full_time[indices]
        rms_group = rms_means[i]
        sigma_group = sigma_means[i]
        #rms_group_sindy = rms_means_sindy[i]
        #sigma_group_sindy = sigma_means_sindy[i]
        #print("Dimensione di valid_t:", len(state_true["valid_t"]))
        #print("Dimensione di rms_group:", len(rms_group))
        #print("Dimensione di sigma_group:", len(sigma_group))

        #plt.plot(data["valid_t"], rms_group, yerr=sigma_group, fmt='-o', color=colors[i], ecolor=colors[i], capsize=5, label=gruppo[i])
        plt.errorbar(valid_t_segment, rms_group, yerr=sigma_group, fmt='-o', color=colors[i], ecolor=colors[i], capsize=5, label=gruppo[i])

    plt.xlabel('tempo')
    plt.ylabel('RMS')
    plt.title('RMS SVISE')
    plt.legend(title=f'SNR: {SNR}')
    plt.grid(True)

    # Definisci il nome del file includendo parametri dinamici
    file_path = f"{folder_path}/confronti_forward_SVISE_{tempo}_{num_dati}_{SNR}.png"
    plt.savefig(file_path, bbox_inches='tight')
    #plt.show()
    del rms_group
    del sigma_group


    full_time = state_true["valid_t"]  # Array completo dei tempi
    ###num_points = [len(rms_means_sindy[i]) for i in range(3)]
    plt.figure(figsize=(10, 6))
    for i in range(3):  # Itera per i 3 gruppi

        interval = len(full_time) / num_points[i]
        indices = np.round(np.arange(0, len(full_time), interval)).astype(int)
        valid_t_segment = full_time[indices]
        ###rms_group_sindy = rms_means_sindy[i]
        ###sigma_group_sindy = sigma_means_sindy[i]


        ###plt.errorbar(valid_t_segment, rms_group_sindy, yerr=sigma_group_sindy, fmt='-o', color=colors[i], ecolor=colors[i], capsize=5, label=gruppo[i])

    plt.xlabel('tempo')
    plt.ylabel('RMS')
    ###plt.title('RMS SINDy')
    plt.legend(title=f'SNR: {SNR}')
    plt.grid(True)

    # Definisci il nome del file includendo parametri dinamici
    ###file_path = f"{folder_path}/confronti_forward_SINDy_{tempo}_{num_dati}_{SNR}.png"
    ###plt.savefig(file_path, bbox_inches='tight')
    #plt.show()
    ###del rms_group_sindy
    ###del sigma_group_sindy




    plt.figure(figsize=(10, 6))
    gruppo = ["SNR 10", "SNR 20", "SNR 100"]
    num_points = [len(rms_means_1[i]) for i in range(3)]
    full_time = state_truee["valid_t"]

    for i in range(3):  # Itera per i 3 gruppi
        interval = len(full_time) / num_points[i]
        indices = np.round(np.arange(0, len(full_time), interval)).astype(int)
        valid_t_segment = full_time[indices]
        rms_group_1 = rms_means_1[i]
        sigma_group_1 = sigma_means_1[i]
        ###rms_group_sindy_1 = rms_means_sindy_1[i]
        ###sigma_group_sindy_1 = sigma_means_sindy_1[i]
        #print("Dimensione di valid_t:", len(state_truee["valid_t"]))
        #print("Dimensione di rms_group_1:", len(rms_group_1))
        #print("Dimensione di sigma_group_1:", len(sigma_group_1))

        #plt.plot(data["valid_t"], rms_group, yerr=sigma_group, fmt='-o', color=colors[i], ecolor=colors[i], capsize=5, label=gruppo[i])
        plt.errorbar(valid_t_segment, rms_group_1, yerr=sigma_group_1, fmt='-o', color=colors[i], ecolor=colors[i], capsize=5, label=gruppo[i])

    plt.xlabel('tempo')
    plt.ylabel('RMS')
    plt.title('RMS SVISE')
    plt.legend(title=f'SNR: {SNR}')
    plt.grid(True)

    # Definisci il nome del file includendo parametri dinamici
    file_path = f"{folder_path}/confronti_forward_SVISE_1s_{tempo}_{num_dati}_{SNR}.png"
    plt.savefig(file_path, bbox_inches='tight')
    #plt.show()
    del rms_group_1
    del sigma_group_1


    full_time = state_truee["valid_t"]  # Array completo dei tempi
    ###num_points = [len(rms_means_sindy_1[i]) for i in range(3)]
    plt.figure(figsize=(10, 6))
    for i in range(3):  # Itera per i 3 gruppi
        interval = len(full_time) / num_points[i]
        indices = np.round(np.arange(0, len(full_time), interval)).astype(int)
        valid_t_segment = full_time[indices]
        ###rms_group_sindy_1 = rms_means_sindy_1[i]
        ###sigma_group_sindy_1 = sigma_means_sindy_1[i]
        

        ###plt.errorbar(valid_t_segment, rms_group_sindy_1, yerr=sigma_group_sindy_1, fmt='-o', color=colors[i], ecolor=colors[i], capsize=5, label=gruppo[i])

    plt.xlabel('tempo')
    plt.ylabel('RMS')
    ###plt.title('RMS SINDy')
    plt.legend(title=f'SNR: {SNR}')
    plt.grid(True)

    # Definisci il nome del file includendo parametri dinamici
    ###file_path = f"{folder_path}/confronti_forward_SINDy_1s_{tempo}_{num_dati}_{SNR}.png"
    ###plt.savefig(file_path, bbox_inches='tight')
    #plt.show()
    ###del rms_group_sindy_1
    ###del sigma_group_sindy_1

    