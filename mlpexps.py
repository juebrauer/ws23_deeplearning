print("loading module mlpexps.py")

import torch
import torch.nn as nn
import torch.optim as optim
import pandas

def prepare_data(fname, list_target_cols):
    print("preparing data ...")
    
    # 1. Bereinigten Datensatz einlesen    
    t = pandas.read_csv(fname)
    
    # 2. Codierung kategorialer Merkmale
    t = pandas.get_dummies(t)
    
    # 3. Input-/Output-Split
    x = t.drop(list_target_cols, axis="columns")
    y = t[  list_target_cols  ]
    
    # 4. Train-/Test-Split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
    
    # 5. Skalierung der Daten
    from sklearn.preprocessing import StandardScaler
    scaler_input = StandardScaler()
    scaler_output = StandardScaler()
    x_train = scaler_input.fit_transform(x_train)
    y_train = scaler_output.fit_transform(y_train)
    x_test = scaler_input.transform(x_test)
    y_test = scaler_output.transform(y_test)
    
    # 6. NumPy-Arrays --> PyTorch Tensoren umwandeln    
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, scaler_input, scaler_output


def create_model(input_dim, output_dim):
    print("creating model ...")
    
    class MLP(nn.Module):

        def __init__(self, input_dim, output_dim):
            super(MLP, self).__init__()
    
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
    
        def forward(self, x):
            return self.layers(x)

    model = MLP(input_dim, output_dim)
    return model


def train_and_test_MLP(epochs, model, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, scaler_output):    
    print("training model ...")
        
    learning_rate = 0.001
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    

    epoch_nr = []
    MAPEs_train = []
    MAPEs_test = []
    for epoch in range(1,epochs+1):
        model.train()
        optimizer.zero_grad()    
        outputs = model(x_train_tensor)    
        l = loss(outputs, y_train_tensor)    
        l.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            MAPE_train = test_MLP("train data", model, x_train_tensor, y_train_tensor, scaler_output)
            MAPE_test  = test_MLP("test data", model, x_test_tensor,  y_test_tensor,  scaler_output)
            print(f"Epoch {epoch}: train MAPE={MAPE_train:.2f}, test MAPE = {MAPE_test:.2f}")
            MAPEs_train.append( MAPE_train )
            MAPEs_test.append( MAPE_test )     
            epoch_nr.append( epoch )
    
    import matplotlib.pyplot as plt
    plt.plot( epoch_nr, MAPEs_train, color="blue", label="MAPE on train data" )
    plt.plot( epoch_nr, MAPEs_test,  color="red", label="MAPE on test data" )
    plt.xlabel("Epoche")
    plt.ylabel("MAPE [%]")
    plt.title("Fehlerkurve / Trainingskurve")
    plt.legend()
    plt.show()


def test_MLP(datatype, model, x_test_tensor, y_test_tensor, scaler_output):
    print(f"\ttesting model on {datatype} ...")
    
    model.eval()
    with torch.no_grad():
        preds = model(x_test_tensor)    
    preds = scaler_output.inverse_transform( preds )
    gt = scaler_output.inverse_transform( y_test_tensor )
    
    from sklearn.metrics import mean_absolute_percentage_error
    MAPE = mean_absolute_percentage_error(gt, preds)  * 100.0    
    return MAPE
    