# Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos que se van a utilizar
df = pd.read_csv('WorldHappinessReport2015_16_18.csv')
# Se les hace un shuffle a los datos 
df = df.sample(frac = 1, random_state = 9).reset_index()
# Se divide el dataframe en variables independientes y en variables dependientes (aquella que se va a predecir) 
df_x = df[['Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity']]
df_y = df[["Happiness Score"]]
# Se agrega una columna , la cual representa a nuestra constante, o valor b
df_x['Constant'] = np.ones(len(df))

# Se divien los dataframes en datos para entrenamiento (90%) y datos para pruebas (10%)
x_train = df_x[:285]
y_train = df_y[:285]
x_test = df_x[285:]
y_test = df_y[285:]

def hyp(params, samples):
  # Función que nos ayuda a sacar la hipótesis, evaluando hyp con los parámetros actuales y las variables independientes almacenadas en x_train
  nparams = np.transpose(params)
  acum= np.dot(nparams,samples)
  return(acum)

def show_errors(params, samples, y):
	# Muestra el error generado con los valores obtenidos en la función de hyp comparados con los valores reales de <y>
	global __errors__
	error_acum = 0
	for i in range(len(samples)): #158 filas
		h = hyp(params, samples.iloc[i])
		print("hyp  %f  y %f " % (h,  y.iloc[i]))
		error = h - y["Happiness Score"].iloc[i]
		error_acum =+ error**2 # función de costo
	mean_error_param = error_acum/len(samples)
	__errors__.append(mean_error_param) # guarda la media de los errores generados
	
def GD(params, samples, y, alfa):
	# Función de gradiente descendiente para enconstrar los nuevos parámetros
	temp = params
	for j in range(len(params)):
		acum =0
		for i in range(len(samples)):
			error = hyp(params,samples.iloc[i]) - y["Happiness Score"].iloc[i] # Diferencia del valor sacado por la hipótesis - eñ valor de <y> real
			acum = acum + error*samples.iloc[i][j]  # Sumatoria parte de la formula de GD para la regresión lineal
			temp[j] = params[j] - alfa*(1/len(samples))*acum  # Resto de la fórmula de GD
	return temp

# Variable global que guarda el error/loss
__errors__= []

# Se definen los parámetros con los que se va a iniciar, samples, el cual es el df de las x para el entrenamiento, y la <y> que es la salida  
params = [0,0,0,0,0,0,0]
samples = x_train
y = y_train

alfa =.01 #learning rate
epochs = 0

while True:  # Corre la función de GD hasta haber completado 5 epocas
	print (params)
	params = GD(params, samples, y, alfa)
	show_errors(params, samples, y)  # Muestra el error
	epochs = epochs + 1
	if(epochs == 6): 
		print ("samples:")
		print(samples)
		print ("final params:")
		print (params)
		break

import matplotlib.pyplot as plt  # Grafia de los errores
plt.plot(__errors__)

# Se utiliza el x_test para realizar predicciones usando los nuevos parámetros
hyp_params = np.transpose(params)
hyp_test = np.transpose(x_test)
hyp_y = np.dot(hyp_params, hyp_test)

# Se almacenan y muestran la comparación entre los valores simulados de <y> y los reales
testing = pd.DataFrame()
testing["Hyp y"] = hyp_y
testing["Real y"] = y_test["Happiness Score"].values
print("Comparación del valor simulado vs el real")
print(testing)
testing.plot() # Gráfica de la comparación 

# Grafico de la comparación a modo de diagrama de dispersión sobre la misma gráfica
x = np.array(range(0,len(x_test-1)))
plt.scatter(x, testing["Hyp y"].values)
plt.scatter(x, testing["Real y"].values)
plt.title("Hyp y vs y real")
plt.xlabel("Index")
plt.ylabel("Happiness Score")
plt.show()