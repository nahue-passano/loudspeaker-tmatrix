# -*- coding: utf-8 -*-
"""
Author: Nahuel Passano

Loudspeaker simulation in free-field
"""

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QFileDialog
from ui_loudspeaker_sim import Ui_Form
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

class Ventana(QWidget):
    def __init__(self):
        super().__init__()
        
        ui = Ui_Form()
        ui.setupUi(self)
            # Connections
        ui.Re.valueChanged.connect(self.simulate)
        ui.Le.valueChanged.connect(self.simulate)
        ui.Bl.valueChanged.connect(self.simulate)
        ui.Mm.valueChanged.connect(self.simulate)
        ui.Cm.valueChanged.connect(self.simulate)
        ui.Rm.valueChanged.connect(self.simulate)
        ui.diam.valueChanged.connect(self.simulate)
        ui.magnitude.toggled.connect(self.simulate)
        ui.phase.toggled.connect(self.simulate)
        ui.pushButton.clicked.connect(self.export)
            
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.sub = self.figure.add_subplot(111)
        self.sub2 = self.sub.twinx()
        layout = QGridLayout()
        layout.addWidget(self.canvas)
        ui.groupBox.setLayout(layout)
        
        self.ui = ui
        self.show()
        
    def simulate(self):
        ui = self.ui
        
        Re = float(ui.Re.value())
        Le = float(ui.Le.value())*10**(-3)
        Bl = float(ui.Bl.value())
        Mm = float(ui.Mm.value())*10**(-3)
        Cm = float(ui.Cm.value())*10**(-3)
        Rm = float(ui.Rm.value())
        diam = float(ui.diam.value())*10**(-2)
        
        # Re = 6.3                # Resistencia eléctrica de la bobina [R]
        # Le = 1.534 *10**(-3)    # Inductancia de la bobina [H] 
        # Bl = 5.089              # Motor magnetico [N/A]
        # Mm = 12.35 *10**(-3)    # Masa mecánica [kg]
        # Cm = 0.671 *10**(-3)    # Compliancia mecánica [m/N]
        # Rm = 0.745              # Resistencia mecánica [N/(m/s)] = [kg/s]
        # diam = 13.4 *10**(-2)   # Diámetro efectico de radiación [m] 
               
                 
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
        # Graphs
        # =============================================================================
        
                 
            # Magnitude
        if ui.magnitude.isChecked():

            self.sub.clear()      
    
                # Sensitivity
            self.sub.semilogx(f,dB(SPL), linewidth=2, color='blue', label ='Sensitivity')
            self.sub.set_ylabel(r'SPL [$dB$]', fontsize=14, color='blue')
            self.sub.set_ylim(bottom=60)
            
                # Impedance         
            self.sub2.clear()
            self.sub2.semilogx(f,np.abs(Z_in_Imp), linewidth = 2 , color='green', label = 'Impedance')
            self.sub2.set_ylabel(r'Impedance [$\Omega$]', fontsize=14, color='green')
            self.sub2.set_yticks(np.array([round(min(np.abs(Z_in)),1)]).tolist() + np.array([round(Re+Bl**2/Rm,1)]).tolist())
            self.sub2.set_ylim(0.75*round(min(np.abs(Z_in)),2),1.25*round(Re+Bl**2/Rm,1))
            self.sub2.legend(loc = 'upper right')
            
                # Frequency
            self.sub.set_xlabel(r'Frequency [$Hz$]', fontsize=14)
            x_ticks = np.sort(np.array([16,125,250,500,1000,2000,4000,round(fs,1)]))
            self.sub.set_xticks(ticks = x_ticks)
            self.sub.set_xticklabels(x_ticks.tolist())
            self.sub.set_xlim(16,4000)
            self.sub.legend(loc = 'upper left')
            self.sub.grid()
        
            # Phase
        else:
                        
            self.sub.clear()
            self.sub2.clear()
            self.sub2.set_yticklabels(labels=list())
            
                # Sensitivity
            self.sub.semilogx(f,np.angle(SPL)*57.29+90, linewidth=2, color='blue', label ='Sensitivity')
            
                # Impedance
            self.sub.semilogx(f,np.angle(Z_in_Imp,deg=True), linewidth=2, color='green', label='Impedance')
            
                # Frequency
            self.sub.set_ylabel(r'Phase [$º$]', fontsize=14)
            self.sub.set_xlabel(r'Frequency [$Hz$]', fontsize=14)
            x_ticks = np.sort(np.array([16,125,250,500,1000,2000,4000,round(fs,1)]))
            self.sub.set_xticks(ticks = x_ticks)
            self.sub.set_xticklabels(x_ticks.tolist())
            self.sub.set_xlim(16,4000)
            self.sub.set_yticks(ticks = np.array([-180,-90,0,90,180]))
            self.sub.set_yticklabels([r"$-180º$",r"$-90º$",r"$0º$",r"$90º$",r"$180º$"])
            self.sub.legend(loc = 'upper right')
            self.sub.grid()
        
        
        
        self.canvas.draw()
        
    def export(self):
        filename = QFileDialog.getSaveFileName(self,"Exportar",None,".png;;.jpg")
        self.figure.savefig(filename[0]+filename[1])
        

        
if __name__ == '__main__':        
        
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        
    ventana = Ventana()
    app.exec_()