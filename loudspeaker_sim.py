# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 18:41:23 2021

@author: NPass
"""

import numpy as np
import matplotlib.pyplot as plt




# =============================================================================
#  Componentes lineales del altoparlante
# =============================================================================

    # Parlante 1

# Re = 5.8              # Resistencia eléctrica de la bobina [R]
# Le = 1.05 *10**(-3)   # Inductancia de la bobina [H] 
# Bl = 5.6              # Motor magnetico [N/A]
# Mm = 10.6 *10**(-3)   # Masa mecánica [kg]
# Cm = 1.4 *10**(-3)    # Compliancia mecánica [m/N]
# Rm = 1.04             # Resistencia mecánica [N/(m/s)] = [kg/s]
# diam = 13.1 *10**(-2) # Diámetro efectico de radiación [m] 

    # Parlante 2

Re = 6.3                # Resistencia eléctrica de la bobina [R]
Le = 1.534 *10**(-3)    # Inductancia de la bobina [H] 
Bl = 5.089              # Motor magnetico [N/A]
Mm = 12.35 *10**(-3)    # Masa mecánica [kg]
Cm = 0.671 *10**(-3)    # Compliancia mecánica [m/N]
Rm = 0.745              # Resistencia mecánica [N/(m/s)] = [kg/s]
diam = 13.4 *10**(-2)   # Diámetro efectico de radiación [m] 
       
    
def sim_frequency_response(Re,Le,Bl,Mm,Cm,Rm,diam):
    """
    Simulación de la respuesta en frecuencia, impedancia y velocidad de 
    un altoparlante.
    
    Gráficos: 
        - Transferencia (Magnitud) vs Frecuencia.
        - Transferencia (Fase) vs Frecuencia
        - Impedancia (Magnitud) vs Frecuencia.
        - Impedancia (Fase) vs Frecuencia.
        - Velocidad (Magnitud) vs Frecuencia.
        - Velocidad (Fase) vs Frecuencia.
        - Impedancia de radiación normalizada vs k*a
        
    Parámetros
    ----------
    Re : float
        Resistencia eléctrica [R]
    Le : float
        Inductancia de la bobina [H]
    Bl : float
        Motor magnético [N/A] , [T*m]
    Mm : float
        Masa mecánica [kg]
    Cm : float
        Compliancia mecánica [m/N]
    Rm : float
        Resistencia mecánica [N/(m/s)] , [kg/s]
    diam : float
        Diámetro efectivo del diafragma [m]

    """
 
    # =============================================================================
    #  Funciones auxiliares (Bessel, Struve y dB)
    # =============================================================================
    
    def bessel(z):
        bessel_sum = 0
        for k in range(25):
            bessel_i = ((-1)**k * (z/2)**(2*k+1)) / (np.math.factorial(k) * np.math.factorial(k+1))
            bessel_sum = bessel_sum + bessel_i
        return bessel_sum
    
    def struve(z):
        struve_sum = 0
        for k in range(25):
            struve_i = (((-1)**k * (z/2)**(2*k+2))) / (np.math.factorial(int(k+1/2)) * np.math.factorial(int(k+3/2)))
            struve_sum = struve_sum + struve_i
        return struve_sum    
    
    def dB(x):
        deciBel = 20*np.log10(np.abs(x))
        return deciBel
    
    # =============================================================================
    #  Constantes acústicas y parametros de T-S.
    # =============================================================================
    
    n=2000
    f = np.logspace(np.log10(10),np.log10(4000),num=n) #Vector de frecuencias
    w = 2*np.pi*f
    
    Ac_c = 343                                  # Velocidad del sonido [m/s]
    Ac_rho0 = 1.16                              # Densidad del aire [kg/m^3]
    Ac_p0 = 101325                              # Presión atmosférica [Pa]
    Ac_p_ref = 20*10**(-6)                      # Presión de referencia [Pa]
    Ac_Zs = Ac_rho0*Ac_c                        # Impedancia característica del aire
    
    k = w/Ac_c                                  # Número de onda
    a = diam/2                                  # Radio efectivo del diafragma
    Sd = np.pi*a**2                             # Superficie efectiva de radiación
    ka = k*a
    f_ka_1 = Ac_c / (2*np.pi * a)               # Frecuencia donde k*a=1
    
    fs = 1 / (2*np.pi*(Mm*Cm)**(1/2))           # Frecuencia de resonancia mecánica
    Qm = 2*np.pi*fs*(Mm+0.00092)/Rm             # Factor de amortiguación mecánico
    Qe = 2*np.pi*fs*(Mm+0.00092)/(Bl**2/Re)     # Factor de amortiguación eléctrico
    Qt = (Qm*Qe) / (Qm+Qe)                      # Factor de amortiguación total
    Vas = (Cm * Sd**2 * Ac_rho0 * Ac_c**2)*1000 # Volumen acústico de la suspensión
    eta = (9.68*10**(-7) * fs**3 * Vas/Qe)      # Rendimiento 
    
    # =============================================================================
    #  Matrices de los elementos mecánicos.
    # =============================================================================
    
    T_Re = np.array([[1 , Re ],                 # Resistencia eléctrica de la bobina.
                     [0 , 1  ]])
    
    T_Le = np.array([[1 , 1j*w*Le ],            # Inductancia de la bobina.
                     [0 , 1       ]])    
    
    G_Bl = np.array([[0    , Bl ],              # Girador de factor Bl.
                     [1/Bl , 0  ]]) 
    
    T_Mm = np.array([[1 , 1j*w*Mm ],            # Masa mecánica.
                     [0 , 1       ]]) 
    
    T_Cm = np.array([[1 , 1/(1j*w*Cm) ],        # Compliancia mecánica.
                     [0 , 1           ]]) 
    
    T_Rm = np.array([[1 , Rm ],                 # Resistencia mecánica.
                     [0 , 1  ]])
    
        # Matriz del motor mecánico.
    T_motor = np.dot(np.dot(np.dot(np.dot(np.dot(T_Re,T_Le),G_Bl),T_Mm),T_Cm),T_Rm)
    
    T_Sd = np.array([[Sd , 0    ],              # Transformador de factor Sd
                     [0  , 1/Sd ]])
    
    # =============================================================================
    #  Impedancia de radiación transferida al mundo mecánico para un pistón plano
    #  de pantalla infinita
    # =============================================================================
    
    ZM_rad_real = np.pi*(a**2)*Ac_Zs*(1-bessel(2*ka)/ka) 
    ZM_rad_imag = np.pi*(a**2)*Ac_Zs*(1j*(struve(2*ka)/ka))
    ZM_rad = ZM_rad_real + ZM_rad_imag
    
    T_ZM = np.array([[1 , ZM_rad],
                     [0 , 1     ]])
    
    """
        ZM_rad: valor cuantitativo de cómo el medio (aire) reacciona contra el 
        movimiento de una superficie vibrante. Se modela en el dominio mecánico 
        como una impedancia en serie.
    """
        
    # =============================================================================
    #  Impedancia acústica especifica para una onda esferica
    # =============================================================================
    
    d=1                                         # Distancia de medición [m]
    Q=2                                         # Factor de directividad
    
    ZA_rad = (1j * w * Ac_rho0 * Q)/(4*np.pi * d * np.exp(1j*k*d))
    Z_delay = 1j*np.exp(-1j * k * d)            # Giro de fase debido a la propagación en el aire
    
    """
        ZA_rad: Relación entre presión y velocidad volumétrica para una onda 
        esférica propagándose por el aire a una distancia 'd'.
    """
    
    # =============================================================================
    #  Matriz del altoparlante incluyendo la parte acústica
    # =============================================================================
    
    T_total = np.dot(np.dot(T_motor,T_ZM),T_Sd)
    
    # =============================================================================
    #  Respuesta en impedancia      
    # =============================================================================
        
        # Inicializador de variables.
    Z_in_Imp = np.zeros(n,dtype = 'complex')
    Z_io_Imp = np.zeros(n,dtype = 'complex')
    I_in_Imp = np.zeros(n,dtype = 'complex')
    out_Imp = np.zeros((2,n),dtype = 'complex')
    
    V_in_Imp = 1                                    # Tensión de entrada.
    
        # Componentes del vector del motor.
    A_Imp = T_motor[0,0]
    B_Imp = T_motor[0,1]
    C_Imp = T_motor[1,0]
    D_Imp = T_motor[1,1]
    
    
    for i in range(n):
        det = np.abs(A_Imp[i]*D_Imp[i]-B_Imp[i]*C_Imp[i])
        T_inv = np.array([[  D_Imp[i]/det , -B_Imp[i]/det ],
                          [ -C_Imp[i]/det ,  A_Imp[i]/det ]])
    
        Z_in_Imp[i] = B_Imp[i] / D_Imp[i]                     # Impedancia de entrada al aire libre (Z_L=0).
        Z_io_Imp[i] = A_Imp[i] / C_Imp[i]                     # Impedancia de entrada con carga (Z_L=inf).
        I_in_Imp = V_in_Imp / Z_in_Imp[i]                     # Corriente de entrada.
        
            # Velocidad y fuerza en función de la frecuencia.
        out_Imp[:,i] = np.dot(T_inv , np.array([V_in_Imp,I_in_Imp]))
        
        
    # =============================================================================
    #  Respuesta en frecuencia
    # =============================================================================
    
        # Inicializador de variables
    Z_L = np.zeros(n,dtype = 'complex')          
    Z_in = np.zeros(n,dtype = 'complex')         
    Z_out = np.zeros(n,dtype = 'complex')        
    H = np.zeros(n,dtype = 'complex')
    Z_rad = np.zeros(n,dtype = 'complex')
    SPL = np.zeros(n,dtype = 'complex')

    
    ###
    
    V_in = 2.83                                 # Tensión de entrada
    
    Z_L = ZA_rad                                # Impedancia de carga Z_L
    
            # Componentes del vector del motor .
    A = T_total[0,0]
    B = T_total[0,1]
    C = T_total[1,0]
    D = T_total[1,1]
    
    for i in range(n):
        
        Z_in[i] = B[i]/D[i]                     # Impedancia de entrada al aire libre (Z_L=0).
        Z_out[i] = D[i]/C[i]                    # Z_out con Z_L = inf.
        H[i] = Z_L[i] / (A[i]*Z_L[i]+B[i])      # Función de transferencia entre voltaje y presión.

        # Cálculo de la respuesta en frecuencia en SPL.
    SPL =  V_in*( H / Z_delay ) / Ac_p_ref
    Z_rad = ZM_rad / (np.pi * a**2 * Ac_Zs)     # Impedancia de radiación.
    
    f_100_index = np.argmin(np.abs(f-100))      # Indice de f para 100 Hz
    f_500_index = np.argmin(np.abs(f-500))      # Indice de f para 1 kHz
    
    sens = np.mean(dB(SPL[f_100_index:f_500_index]))
    # =============================================================================
    # Graficos
    # =============================================================================
    
    
    fig = plt.figure(figsize=(72, 45))
    plt.subplots_adjust(hspace=0.3,wspace=0.2)

    title = 'Parámetros del altoparlante'
    plt.figtext(0.15,0.85,s=title,fontsize=55,fontweight='bold')
    
    sens_text = '-Sensibilidad: ' + str(round(sens,1)) + ' $dB_{SPL}$'
    plt.figtext(0.171,0.81,s=sens_text,fontsize=50)
    
    fs_text = '- $f_s$: ' + str(round(fs,1)) + ' Hz'
    plt.figtext(0.17,0.78,s=fs_text,fontsize=50)
    
    Qm_text = '- $Q_m$: ' + str(round(Qm,2))
    plt.figtext(0.17,0.755,s=Qm_text,fontsize=50)
    
    Qe_text = '- $Q_e$: ' + str(round(Qe,2))
    plt.figtext(0.17,0.73,s=Qe_text,fontsize=50)
    
    Qt_text = '- $Q_t$: ' + str(round(Qt,2))
    plt.figtext(0.17,0.705,s=Qt_text,fontsize=50)
    
    Re_text = '- $R_e$: ' + str(round(Re,1)) + ' $\Omega$'
    plt.figtext(0.235,0.78,s=Re_text,fontsize=50)
    
    Bl_text = '- $Bl$: ' + str(round(Bl,2)) + ' $Tm$'
    plt.figtext(0.235,0.755,s=Bl_text,fontsize=50)
    
    Vas_text = '- $V_{as}$: ' + str(round(Vas,2)) + ' $l$'
    plt.figtext(0.235,0.73,s=Vas_text,fontsize=50)
    
    eta_text = '- $\eta$: ' + str(round(eta,2)) + '$\%$'
    plt.figtext(0.235,0.705,s=eta_text,fontsize=50)
    
    ka1_text = '(Simulación válida hasta $f_{ka=1} = $' + str(round(f_ka_1,1)) + ' Hz)' 
    plt.figtext(0.155,0.67,s=ka1_text,fontsize=40,color='grey')
    
        # Frecuencia (Magnitud)
        
    plt.subplot2grid((3,3),(0,1))
    plt.semilogx(f,dB(SPL), linewidth=8, color='blue')
    plt.title(r'Respuesta en frecuencia del altoparlante', fontsize=40,fontweight="bold")
    plt.ylabel(r'Presión [$dB_{SPL}$]', fontsize=45)
    plt.xlabel('Frecuencia [Hz]',fontsize=45)
    x_ticks = np.sort(np.array([16,125,250,500,1000,2000,4000,round(fs,1)]))
    plt.xticks(ticks = x_ticks,labels = x_ticks.tolist() ,fontsize=30, rotation=45)
    plt.yticks(fontsize=38) 
    plt.xlim(10,4000)
    plt.grid()
    
        # Frecuencia (Fase)
    
    plt.subplot2grid((3,3),(0,2))
    plt.semilogx(f,np.angle(SPL)*57.29+90, linewidth=8, color='blue')
    plt.title(r'Respuesta en fase del altoparlante', fontsize=40,fontweight="bold")
    plt.ylabel(r'Fase [$º$]', fontsize=45)
    plt.xlabel('Frecuencia [Hz]',fontsize=45)
    x_ticks = np.sort(np.array([16,125,250,500,1000,2000,4000,round(fs,1)]))
    plt.xticks(ticks = x_ticks,labels = x_ticks.tolist() ,fontsize=30, rotation=45)
    y_label1 = [r"$-180º$",r"$-90º$",r"$0º$",r"$90º$",r"$180º$"]
    plt.yticks(np.array([-180,-90,0,90,180]),y_label1,fontsize=38)
    plt.xlim(10,4000)
    plt.ylim(-180,180)
    plt.grid() 
    
    
        # Impedancia (Magnitud)
        
    plt.subplot2grid((3,3),(1,1))
    plt.semilogx(f,np.abs(Z_in_Imp), linewidth=8, color='green')
    plt.title(r'Respuesta en magnitud de la impedancia eléctrica', fontsize=40,fontweight="bold")
    plt.ylabel(r'Impedancia [$\Omega$]', fontsize=45)
    plt.xlabel('Frecuencia [Hz]',fontsize=45)
    x_ticks = np.sort(np.array([16,125,250,500,1000,2000,4000,round(fs,1)]))
    plt.xticks(ticks = x_ticks,labels = x_ticks.tolist() ,fontsize=30,rotation=45)
    plt.yticks(np.array([round(min(np.abs(Z_in)),2)]).tolist() + np.array([round(Re+Bl**2/Rm,1)]).tolist(),fontsize=38) 
    plt.xlim(10,4000)
    plt.grid()
    
        # Impedancia (Fase)
    
    plt.subplot2grid((3,3),(1,2))
    plt.semilogx(f,np.angle(Z_in_Imp,deg=True), linewidth=8, color='green')
    plt.title(r'Respuesta en fase de la impedancia eléctrica', fontsize=40,fontweight="bold")
    plt.ylabel(r'Fase [$º$]', fontsize=45)
    plt.xlabel('Frecuencia [Hz]',fontsize=45)
    x_ticks = np.sort(np.array([16,125,250,500,1000,2000,4000,round(fs,1)]))
    plt.xticks(ticks = x_ticks,labels = x_ticks.tolist() ,fontsize=30, rotation=45)
    y_label1 = [r"$-180º$",r"$-90º$",r"$0º$",r"$90º$",r"$180º$"]
    plt.yticks(np.array([-180,-90,0,90,180]),y_label1,fontsize=38)
    plt.xlim(10,4000)
    plt.ylim(-180,180)
    plt.grid()
    
        # Velocidad (Magnitud)
        
    plt.subplot2grid((3,3),(2,1))
    plt.semilogx(f,np.abs(out_Imp[1,:]), linewidth=8,color='orange')
    plt.title(r'Respuesta en magnitud de la velocidad mecánica', fontsize=40,fontweight="bold")
    plt.ylabel(r'Velocidad [$\frac{m}{s}$]', fontsize=45)
    plt.xlabel('Frecuencia [Hz]',fontsize=45)
    x_ticks = np.sort(np.array([16,125,250,500,1000,2000,4000,round(fs,1)]))
    plt.xticks(ticks = x_ticks,labels = x_ticks.tolist() ,fontsize=30, rotation=45)
    plt.yticks(list(plt.yticks()[0]) + np.array([round(max(np.abs(out_Imp[1,:])),3)]).tolist(),fontsize=38) 
    plt.xlim(10,4000)
    plt.ylim(bottom=0)
    plt.grid()
    
        # Velocidad (Fase)
    
    plt.subplot2grid((3,3),(2,2))
    plt.semilogx(f,np.angle(-out_Imp[1,:],deg=True), linewidth=8,color='orange')
    plt.title(r'Respuesta en fase de la velocidad mecánica', fontsize=40,fontweight="bold")
    plt.ylabel(r'Fase [$º$]', fontsize=45)
    plt.xlabel('Frecuencia [Hz]',fontsize=45)
    x_ticks = np.sort(np.array([16,125,250,500,1000,2000,4000,round(fs,1)]))
    plt.xticks(ticks = x_ticks,labels = x_ticks.tolist() ,fontsize=30, rotation=45)
    y_label1 = [r"$-180º$",r"$-90º$",r"$0º$",r"$90º$",r"$180º$"]
    plt.yticks(np.array([-180,-90,0,90,180]),y_label1,fontsize=38)
    plt.xlim(10,4000)
    plt.ylim(-180,180)
    plt.grid()
    
        # Impedancia de radiación
        
    plt.subplot2grid((3,3),(1,0),rowspan=2)
    plt.semilogx(ka,dB(((Z_rad.tolist()))), linewidth=16, color = 'black',label='$Z_{rad}$')
    plt.semilogx(ka,dB(np.real(((Z_rad.tolist())))), linewidth=8, color = 'green', linestyle='dashed',label='$Re\{Z_{rad}\}$')
    plt.semilogx(ka,dB(np.imag(((Z_rad.tolist())))), linewidth=8, color = 'red', linestyle='dashed',label='$Im\{Z_{rad}\}$')
    plt.title(r'Impedancia de radiación', fontsize=40,fontweight="bold")
    plt.ylabel(r'Impedancia normalizada', fontsize=45)
    plt.xlabel('$k\cdot a$ ',fontsize=45)
    x_ticks = np.array([0.01, 0.1, 1, 10])
    x_labels = ['0.01', '0.1', '1', '10']
    plt.xticks(ticks = x_ticks,labels=x_labels,fontsize=38)
    plt.yticks(fontsize=38)
    plt.xlim(0,10)
    plt.ylim(top=10)
    plt.legend(fontsize='xx-large',loc = 'upper left',prop={'size':40})
    plt.grid()
    
    
    plt.show()
    
    question = input('¿Desea guardar el gráfico? [Y/N]: ')
    
    if question == 'Y':
        name = input('Nombre con el que desea guardar el gráfico: ')
        graph_name = name + '.png'
        fig.savefig(graph_name)
        

sim_frequency_response(Re,Le,Bl,Mm,Cm,Rm,diam)





















