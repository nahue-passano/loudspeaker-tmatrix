# Loudspeaker response simulation (free-field)

In a Electroacoustics course, a simulation was developed in Python that allows to study loudspeakers response in frequency, impedance and velocity with some Thiele-Small parameters as input. 

The notebook has six parts:

- **1. Libraries and auxiliar functions:** Loads the libreries needed (MatPlotLib and Numpy) and defines the Bessel and Struve functions requiered for describe the radiaton impedance of the air
- **2. Thiele-Small parameters:** Set the linear parameters of the loudspeaker. In fact, with only the electric resistance and inductance of the coil ($R_e$,$L_e$), the mechanical mass of the diaphragm ($M_{M_D}$), the mechanical compliance and resistance of the suspension ($C_{M_S}$,$R_{M_S}$), the $Bl$ factor and the effective diameter of the cone, the rest of parameters are calculated.



