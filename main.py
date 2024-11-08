import thermal_process as tp
import matplotlib.pyplot as plt

# Constantes y parámetros
R = 8.314  # Constante de gas ideal, J/(mol*K)
n = 1      # Cantidad de sustancia, mol
T_hot = 1000  # Temperatura de alta (K)
T_cold = 430 # Temperatura de baja (K)
Vi = 30    # Volumen inicial (m^3)
VMax = 90    # Volumen en expansión isotérmica (m^3)
p_iso = 1000 #presion isobara



tp.carnot_igas(n,Vi,VMax,T_cold,T_hot)

tp.otto_igas(n,Vi,VMax,T_cold,T_hot)

tp.diesel_igas(n,Vi,VMax,T_hot,p_iso)
