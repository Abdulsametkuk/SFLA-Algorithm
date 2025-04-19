#Kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

# Küresel parametreler
memeplexes_number = 25  # Memeplex sayısı
frog_n = 10  # Her memeplex'teki kurbağa sayısı
total = memeplexes_number * frog_n  # Toplam populasyon
circulation_N = 10  # Dolaşım sayısı
submemep_q = 5  # Alt memeplex eleman sayısı
total_eval = 200  # Toplam iterasyon sayısı

# Veri seti yükleme ve hazırlama
data = pd.read_csv("C:/Users/apoku/Desktop/Japon_SFLA/winequality-red.csv")  # Veri seti yolu
X = data.iloc[:, :-1].values   #özellik sütunlarını indeksleme 
y = data.iloc[:, -1].values    

# Eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelin tahmin fonksiyonu
def predict(parameters, X): #Ağırlıklar ve Bağımsız Değişkenler
    return np.dot(X, parameters[:-1]) + parameters[-1]

# Uygunluk (Fitness) Fonksiyonu
def mseFitness(parameters):
    predictions = predict(parameters, X_train)
    return mean_squared_error(y_train, predictions)

# Popülasyon başlatma
def initialPopula(dim, low, up): #Popülasyondaki her kurbağanın alabileceği en az ve en fazla değer 
    return np.random.uniform(low, up, size=(total, dim)) #her kurbağa rastgele ve belirtilen sınırlar arasında konumlandırılır.

# Fitness hesaplama
def calculateFitness(func, params):
    return np.apply_along_axis(func, 1, params) #Numoy dizisi kullanılarak her kurbağanın fitness değerini hesaplanır.

# Memeplex oluşturma
def createMemeplexes(populations):
    return populations.reshape(memeplexes_number, frog_n, -1) #popülasyon üzerinden memeplexler oluşturulur.

# Lokal arama
def localSearch(memeplexes, func, global_best):
    for im in range(memeplexes_number): 
        for _ in range(circulation_N): #Her memeplex için dolaşım sayısı kadar yerel arama yapılır.
            submemep = constructSubmemep(memeplexes[im]) #Alt memeplex oluşturulur.
            sub_fitness = calculateFitness(func, submemep) #Alt memeplexin fitness değerleri hesaplanır.

            sorted_indices = np.argsort(sub_fitness) #Fitness değerleri sıralanır.
            submemep = submemep[sorted_indices] #Fitness değerleri en iyi olan kurbağa ile en kötü olan kurbağa arasında yer değitirir
            sub_best = submemep[0] #En iyi kurbağa
            sub_worst = submemep[-1] #En kötü kurbağa
            new_position = updateWorst(sub_best, sub_worst, global_best) #En kötü kurbağa en iyi kurbağa ile değiştirilir.
            memeplexes[im][-1] = new_position #Memeplexin en son elemanı güncellenir.
    return memeplexes.reshape(-1, memeplexes.shape[-1]) #Memeplexler birleştirilir.

# Alt memeplex oluşturma
def constructSubmemep(current_memep): 
    indices = np.random.choice(frog_n, submemep_q, replace=False)  #Alt memeplex elemanları rastgele olarak seçilir.
    return current_memep[indices] 

# En kötü pozisyonu güncelleme
def updateWorst(local_best, local_worst, global_best): 
    step = random.random() * (local_best - local_worst) 
    new_position = local_worst + step
    return new_position

# Global arama ve animasyon
def globalSearch(func, dimension, low_limit=-10, up_limit=10): 
    populations = initialPopula(dimension, low_limit, up_limit) 
    best_fitness_history = []  # Fitness geçmişini saklamak için liste

    # Dinamik renkler
    colors = cm.get_cmap('tab20', memeplexes_number).colors 

    # Animasyon için hazırlık
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # 1 satır, 2 sütun
    scatters = []
    for i in range(memeplexes_number):
        scatter = ax1.scatter([], [], label=f"Memeplex {i+1}", color=colors[i])
        scatters.append(scatter)
    
    ax1.set_xlim(low_limit, up_limit) 
    ax1.set_ylim(low_limit, up_limit)
    ax1.set_title("Kurbağa Konumlarının Optimizasyonu")
    ax1.set_xlabel("Parametre 1")
    ax1.set_ylabel("Parametre 2")
    ax1.legend()

    def update(frame):
        nonlocal populations 
        fitness = calculateFitness(func, populations) 
        sorted_indices = np.argsort(fitness)
        populations = populations[sorted_indices]
        best_fitness = fitness[0]
        best_fitness_history.append(best_fitness)  # En iyi fitness değerini kaydet

        memeplexes = createMemeplexes(populations)
        populations = localSearch(memeplexes, func, populations[0])

        # Scatter plot güncelleme
        for i, scatter in enumerate(scatters):
            scatter.set_offsets(memeplexes[i][:, :2])
        
        # Fitness evrim grafiğini güncelleme
        ax2.clear()  # Önceki grafiği temizle
        ax2.plot(range(len(best_fitness_history)), best_fitness_history, color='b', label='Best Fitness (MSE)')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness (MSE)')
        ax2.set_title('Fitness Evolution')
        ax2.legend()

        ax1.set_title(f"Iteration {frame + 1}: Best Fitness = {best_fitness:.4f}")
        return scatters

    ani = FuncAnimation(fig, update, frames=total_eval, repeat=False)

    plt.show()

    # En iyi çözüm ve performans
    fitness = calculateFitness(func, populations)
    best_solution = populations[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    print("Optimized Parameters:", best_solution) # özelliklerin optimize edilmiş ağırlıkları 
    print("Best Fitness (MSE):", best_fitness) 

    # Test set performansı
    test_predictions = predict(best_solution, X_test) 
    test_mse = mean_squared_error(y_test, test_predictions) 
    print("Test Set MSE:", test_mse) 

# Ana program
if __name__ == '__main__':
    parameter_dim = X.shape[1] + 1
    globalSearch(mseFitness, dimension=parameter_dim) 
