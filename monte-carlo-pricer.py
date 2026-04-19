import numpy as np
import scipy.stats as si
import time
import matplotlib.pyplot as plt
from matplotlib import cm

def black_scholes_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calcule le prix exact de l'option Call selon Black-Scholes.
    Utilisé comme benchmark mathématique pour valider le modèle.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S0 * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

def monte_carlo_call_vectorized(S0: float, K: float, T: float, r: float, sigma: float, num_simulations: int = 100_000):
    """
    Pricer Monte Carlo vectorisé pour une option Call Européenne.
    
    Paramètres :
    S0 (float) : Prix actuel de l'actif sous-jacent (Spot)
    K (float) : Prix d'exercice (Strike)
    T (float) : Temps jusqu'à maturité (en années)
    r (float) : Taux d'intérêt sans risque
    sigma (float) : Volatilité de l'actif
    num_simulations (int) : Nombre de trajectoires simulées
    
    Retourne :
    float : Prix estimé de l'option
    float : Temps d'exécution
    """
    start_time = time.time()
    
    # 1. Génération vectorisée de variables normales centrées réduites Z ~ N(0,1)
    Z = np.random.standard_normal(num_simulations)
    
    # 2. Calcul vectorisé des prix à maturité S(T) via le Mouvement Brownien Géométrique
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # 3. Calcul des Payoffs pour un Call : max(S_T - K, 0)
    payoffs = np.maximum(ST - K, 0)
    
    # 4. Actualisation du gain moyen (Discounting)
    call_price = np.exp(-r * T) * np.mean(payoffs)
    
    execution_time = time.time() - start_time
    
    return call_price, execution_time

def monte_carlo_greeks(S0: float, K: float, T: float, r: float, sigma: float, num_simulations: int = 100_000):
    """
    Calcule Delta, Gamma et Vega via la méthode des différences finies centrales.
    Utilise des "Common Random Numbers" (CRN) pour annuler le bruit statistique
    entre les différentes trajectoires bumpées.
    """
    start_time = time.time()
    
    # 1. Génération des CRN (Common Random Numbers) - Mêmes Z pour tous les calculs
    Z = np.random.standard_normal(num_simulations)
    
    # 2. Définition des chocs (Bumps)
    dS = S0 * 0.01      # Choc de 1% sur le prix du sous-jacent (Spot)
    dVol = 0.01         # Choc de 1% en absolu sur la Volatilité
    
    # 3. Fonction interne de pricing rapide (réutilise le même Z)
    def get_price(spot, vol):
        ST = spot * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
        return np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))
    
    # 4. Pricing des scénarios "bumpés"
    P0 = get_price(S0, sigma)                  # Prix central
    P_up = get_price(S0 + dS, sigma)           # Prix Spot + 1%
    P_down = get_price(S0 - dS, sigma)         # Prix Spot - 1%
    
    P_vol_up = get_price(S0, sigma + dVol)     # Prix Volatilité + 1%
    P_vol_down = get_price(S0, sigma - dVol)   # Prix Volatilité - 1%
    
    # 5. Calcul des sensibilités (Différences finies)
    delta = (P_up - P_down) / (2 * dS)
    gamma = (P_up - 2 * P0 + P_down) / (dS ** 2)
    vega = (P_vol_up - P_vol_down) / (2 * dVol) 
    vega_1pct = vega / 100  # Les traders expriment toujours le Vega pour 1% de variation de Vol.
    
    exec_time = time.time() - start_time
    
    return delta, gamma, vega_1pct, exec_time

def plot_monte_carlo_dashboard(S0: float, K: float, T: float, r: float, sigma: float):
    """
    Génère 3 graphiques séparés : 
    1. Trajectoires stochastiques (GBM).
    2. Convergence de l'estimateur.
    3. Distribution des payoffs (Histogramme).
    """
    import matplotlib.pyplot as plt
    
    # --- PRÉPARATION DES DONNÉES ---
    # 1. Trajectoires
    steps = 252 
    dt = T / steps
    paths = np.zeros((steps + 1, 100))
    paths[0] = S0
    for t in range(1, steps + 1):
        Z = np.random.standard_normal(100)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
    # 2. Convergence & Distribution
    N_sims = 50_000 # 50k simulations pour un bel histogramme lisse
    Z_conv = np.random.standard_normal(N_sims)
    ST_conv = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_conv)
    payoffs = np.exp(-r * T) * np.maximum(ST_conv - K, 0)
    
    running_mean = np.cumsum(payoffs) / np.arange(1, N_sims + 1)
    bs_exact_price = black_scholes_call(S0, K, T, r, sigma)

    # Style global
    plt.style.use('dark_background')

    # --- FENÊTRE 1 : LES TRAJECTOIRES ---
    plt.figure(1, figsize=(8, 5))
    plt.plot(np.linspace(0, T, steps + 1), paths, alpha=0.6, linewidth=0.8)
    plt.axhline(y=K, color='r', linestyle='--', label='Strike (K)')
    plt.title('1. Simulation de 100 Trajectoires (GBM)', fontsize=14, fontweight='bold')
    plt.xlabel('Temps (Années)')
    plt.ylabel('Prix du Sous-Jacent')
    plt.legend()
    plt.tight_layout()

    # --- FENÊTRE 2 : LA CONVERGENCE ---
    plt.figure(2, figsize=(8, 5))
    plt.plot(running_mean, color='cyan', label='Prix Monte Carlo')
    plt.axhline(y=bs_exact_price, color='r', linestyle='--', label='Prix Black-Scholes Exact')
    plt.title('2. Convergence de la Simulation', fontsize=14, fontweight='bold')
    plt.xlabel('Nombre de Simulations')
    plt.ylabel('Prix Estimé du Call')
    plt.legend()
    plt.tight_layout()

    # --- FENÊTRE 3 : LA DISTRIBUTION (NOUVEAU) ---
    plt.figure(3, figsize=(8, 5))
    # On affiche l'histogramme avec 100 "bacs" (bins) pour bien voir la répartition
    plt.hist(payoffs, bins=100, color='mediumspringgreen', alpha=0.7, edgecolor='black')
    plt.axvline(x=np.mean(payoffs), color='r', linestyle='dashed', linewidth=2, label=f'Moyenne (Prix): {np.mean(payoffs):.2f} €')
    plt.title('3. Discounted Call Payoff Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Payoff Actualisé (€)')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.tight_layout()

    print("\nGénération des graphiques...")
    print("(3 fenêtres séparées vont s'ouvrir. Fermez-les toutes pour terminer le script.)")
    plt.show()

# ==========================================
# TEST DU MOTEUR
# ==========================================
if __name__ == "__main__":
    # Paramètres de marché fictifs (Option At-The-Money)
    S0_val = 100.0      
    K_val = 100.0       
    T_val = 1.0         
    r_val = 0.05        
    sigma_val = 0.20    
    N_simulations = 1_000_000  
    
    print(f"Lancement de {N_simulations:,} simulations...")
    
    # Calcul via Monte Carlo
    mc_price, exec_time = monte_carlo_call_vectorized(S0_val, K_val, T_val, r_val, sigma_val, N_simulations)
    
    # Calcul via Black-Scholes (Benchmark)
    bs_price = black_scholes_call(S0_val, K_val, T_val, r_val, sigma_val)
    
    # Affichage des résultats
    print("-" * 50)
    print(f"Prix estimé (Monte Carlo) : {mc_price:.5f} € (en {exec_time:.4f} sec)")
    print(f"Prix exact (Black-Scholes): {bs_price:.5f} €")
    print(f"Erreur absolue de pricing : {abs(mc_price - bs_price):.5f} €")
    print("-" * 50)

    # === MODULE DE RISQUE (GREEKS) ===
    print("\nLancement du Risk Engine (Calcul des Greeks)...")
    delta, gamma, vega, greek_time = monte_carlo_greeks(S0_val, K_val, T_val, r_val, sigma_val, N_simulations)
    
    print("-" * 50)
    print(f"Delta (Δ) : {delta:.5f} (Couverture directionnelle)")
    print(f"Gamma (Γ) : {gamma:.5f} (Risque de convexité)")
    print(f"Vega  (ν) : {vega:.5f} € (Impact pour +1% de Volatilité)")
    print(f"Temps d'exécution : {greek_time:.4f} sec")
    print("-" * 50)

    # === MODULE DE VISUALISATION ===
    plot_monte_carlo_dashboard(S0_val, K_val, T_val, r_val, sigma_val)
