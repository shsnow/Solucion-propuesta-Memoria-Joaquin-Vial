import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# Paso 1: Cargar los datos de entrenamiento desde los archivos CSV
train_data_files = ['data/gsk80.csv', 'data/gsk90.csv','data/gsk100.csv']

train_inputs = []
train_outputs = []

# Leer los datos de entrenamiento de cada archivo CSV y almacenarlos en listas
for file in train_data_files:
    data = pd.read_csv(file, delimiter=';').values
    outputs = data[:, 0]  # Primer valor en cada fila es el resultado deseado
    inputs = data[:, 1:]  # Los siguientes 7 valores son los datos de entrada

    train_inputs.append(inputs)
    train_outputs.append(outputs)

# Paso 2: Convertir los datos de entrenamiento en tensores de PyTorch

train_inputs = torch.Tensor(train_inputs)
train_outputs = torch.Tensor(train_outputs)

# Paso 3: Cargar los datos de evaluación desde los archivos CSV
eval_data_files = ['data/gsk85.csv', 'data/gsk95.csv']
eval_inputs = []
eval_outputs = []




# Leer los datos de evaluación de cada archivo CSV y almacenarlos en listas
for file in eval_data_files:
    data = pd.read_csv(file, delimiter=';').values
    outputs = data[:, 0]  # Primer valor en cada fila es el resultado deseado
    inputs = data[:, 1:]  # Los siguientes 7 valores son los datos de entrada

    eval_inputs.append(inputs)
    eval_outputs.append(outputs)

# Paso 4: Convertir los datos de evaluación en tensores de PyTorch
eval_inputs = torch.Tensor(eval_inputs)
eval_outputs = torch.Tensor(eval_outputs)



# Imprimir una muestra de los datos de entrenamiento para asegurarnos de que tomamos bien las columnas
print("Datos de entrenamiento:")
print("Inputs:")
print(train_inputs[0])  # Imprimir el primer ejemplo de inputs
print("Output:")
print(train_outputs[0])  # Imprimir el primer ejemplo de output

# Imprimir una muestra de los datos de evaluación
print("Datos de evaluación:")
print("Inputs:")
print(eval_inputs[0])  # Imprimir el primer ejemplo de inputs
print("Output:")
print(eval_outputs[0])  # Imprimir el primer ejemplo de output


# Paso 5: Definir la arquitectura de la RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, _ = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])

        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return hidden

# Paso 6: Definir los parámetros del modelo
input_size = 7  # Cantidad de características de entrada
hidden_size = 64  # Cantidad de neuronas en la capa oculta
output_size = 1

# Paso 7: Crear una instancia del modelo
model = RNN(input_size, hidden_size, output_size)


num_epochs = 50
learning_rate = 0.07
weight_decay = 0.0001

# Paso 8: Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
#Agregar regularización L2 al optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Paso 9: Entrenamiento del modelo
for epoch in range(num_epochs):
    # Reiniciar los gradientes
    optimizer.zero_grad()

    # Propagación hacia adelante
    outputs = model(train_inputs)

    # Calcular la pérdida
    loss = criterion(outputs, train_outputs)

    # Retropropagación y optimización
    loss.backward()
    optimizer.step()

    # Imprimir la pérdida en cada epoch
    print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, loss.item()))

# Paso 10: Evaluación del modelo en los datos de evaluación
eval_outputs_pred = model(eval_inputs)
eval_loss = criterion(eval_outputs_pred, eval_outputs)
print('Evaluation Loss:', eval_loss.item())

# Imprimir el MSE final
final_mse = criterion(outputs, train_outputs)
print('Final MSE:', final_mse.item())