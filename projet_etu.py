#Habiba ELHUSSIENY 
#Sara ELAMOURI 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):  # On fait hériter MSELoss de Loss
    def forward(self, y, yhat):
        return np.mean((y - yhat) ** 2, axis=1)

    def backward(self, y, yhat):
        return 2 * (yhat - y) / y.shape[1]  # Dérivée partielle de MSE par rapport à yhat

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._parameters = {
            "W": np.random.randn(input_size, output_size) * 0.1,  # Poids initialisés aléatoirement
            "b": np.zeros((1, output_size))  # Biais initialisé à 0
        }
        self._gradient = {
            "W": np.zeros((input_size, output_size)),
            "b": np.zeros((1, output_size))
        }

    def forward(self, X):
        self.input = X  # Sauvegarde de X pour la backpropagation
        return X @ self._parameters["W"] + self._parameters["b"]

    def backward_update_gradient(self, input, delta):
        self._gradient["W"] += input.T @ delta  # Gradient des poids
        self._gradient["b"] += np.sum(delta, axis=0, keepdims=True)  # Gradient du biais

    def backward_delta(self, input, delta):
        return delta @ self._parameters["W"].T  # Propagation du gradient

    def update_parameters(self, gradient_step=1e-3):
        self._parameters["W"] -= gradient_step * self._gradient["W"]
        self._parameters["b"] -= gradient_step * self._gradient["b"]
        self.zero_grad()  # Réinitialisation des gradients
    
    def zero_grad(self):
        for key in self._gradient:
            self._gradient[key].fill(0)  

class TanH(Module):
    def __init__(self):
        super().__init__()  # Hérite correctement de Module

    def forward(self, X):
        self.output = np.tanh(X)  # Sauvegarde la sortie pour backward
        return self.output

    def backward_delta(self, input, delta):
        return delta * (1 - self.output ** 2)  # Dérivée de TanH

    def backward_update_gradient(self, input, delta):
        pass 

    def update_parameters(self, gradient_step):
        pass  

class Sigmoid(Module):
    def __init__(self):
        super().__init__()  # Hérite correctement de Module

    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))  # Sauvegarde la sortie pour backward
        return self.output

    def backward_delta(self, input, delta):
        return delta * (self.output * (1 - self.output))  # Dérivée de la sigmoïde

    def backward_update_gradient(self, input, delta):
        pass  

    def update_parameters(self, gradient_step):
        pass  

class Sequentiel(Module):
    def __init__(self):
        super().__init__()
        self.modules = []

    def add_module(self, module):
        self.modules.append(module)

    def forward(self, X):
        self.inputs = [X]  # pour garder les entrées à chaque étape
        out = X
        for module in self.modules:
            out = module.forward(out)
            self.inputs.append(out)  # stocke la sortie pour backward
        return out

    def backward(self, y, yhat, loss_fn):
        delta = loss_fn.backward(y, yhat)
        for i in reversed(range(len(self.modules))):
            module = self.modules[i]
            input_i = self.inputs[i]
            #D'abord utiliser delta tel quel pour mettre à jour les gradients
            module.backward_update_gradient(input_i, delta)
            #Ensuite calculer le delta à transmettre à la couche précédente
            delta = module.backward_delta(input_i, delta)

    def update_parameters(self, learning_rate):
        for module in self.modules:
            module.update_parameters(learning_rate)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()
    
class Optim:
    def __init__(self, net, loss_fn, eps):
        self.net = net
        self.loss_fn = loss_fn
        self.eps = eps

    def step(self, batch_x, batch_y):
        self.net.zero_grad()
        yhat = self.net.forward(batch_x)
        loss = self.loss_fn.forward(batch_y, yhat)
        self.net.backward(batch_y, yhat, self.loss_fn)
        self.net.update_parameters(self.eps)
        return np.mean(loss)  # utile si on veut suivre l'évolution de la perte

def SGD(optimizer, X_train, y_train, batch_size, epochs, verbose=True):
    N = X_train.shape[0]

    for epoch in range(epochs):
        # Mélange aléatoire des indices
        indices = np.random.permutation(N)

        # Découpage en mini-batchs
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            # Effectuer une étape d'optimisation
            loss = optimizer.step(batch_X, batch_y)

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X_shift = X - np.max(X, axis=1, keepdims=True)  # stabilité numérique
        exp_X = np.exp(X_shift)
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output

    def backward_delta(self, input, delta):
        return delta  # pas besoin de recalculer si on utilise cross-entropy derrière

    def backward_update_gradient(self, input, delta):
        pass  # Pas de paramètres ,rien à faire

    def update_parameters(self, gradient_step):
        pass  # Pas de paramètres à mettre à jour

class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        # Évite log(0) → ajoute un epsilon
        eps = 1e-15
        yhat_clipped = np.clip(yhat, eps, 1 - eps)

        # On applique la formule : -∑ y_i log(yhat_i)
        # Comme y est one-hot, on peut simplement faire : -log(yhat pour la bonne classe)
        loss = -np.sum(y * np.log(yhat_clipped), axis=1)  # un vecteur de taille batch
        return loss  # on peut faire .mean() à l’extérieur si besoin

    def backward(self, y, yhat):
        return yhat - y
    
class LogSoftmax(Module):
    def __init__(self):
        super().__init__()


    def forward(self, X):
        # stabilité numérique
        X_shift = X - np.max(X, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(X_shift), axis=1, keepdims=True))
        self.output = X_shift - log_sum_exp
        return self.output

    
    def backward_delta(self, input, delta):
        return delta  # pareil que Softmax si on combine avec NLLLoss


    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step):
        pass

class NLLLoss(Loss):
    def forward(self, y, log_yhat):
        loss = -np.sum(y * log_yhat, axis=1)
        return loss

    def backward(self, y, log_yhat):
        softmax = np.exp(log_yhat)
        return softmax - y
    

class BCE(Module):
    def forward(self, y, yhat):
        yhat = np.clip(yhat, 1e-7, 1 - 1e-7)  # Evite log(0)
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def backward(self, y, yhat):
        yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
        return (yhat - y) / (yhat * (1 - yhat)) / y.shape[0]


class AutoEncodeur(Module):
    def __init__(self, encodeur, decodeur):
        super().__init__()
        self.encodeur = encodeur
        self.decodeur = decodeur

    def forward(self, x):
        self.encoded = self.encodeur.forward(x)
        self.decoded = self.decodeur.forward(self.encoded)
        return self.decoded

    def backward(self, X, X_hat, loss_fn):
        # Calcul du gradient de la loss par rapport à la sortie décodée
        delta = loss_fn.backward(X, X_hat)

        # Backward sur le décodeur avec X_hat = output reconstruit
        for i in reversed(range(len(self.decodeur.modules))):
            module = self.decodeur.modules[i]
            input_i = self.decodeur.inputs[i]
            module.backward_update_gradient(input_i, delta)
            delta = module.backward_delta(input_i, delta)

        # Backward sur l’encodeur avec le delta propagé du décodeur
        for i in reversed(range(len(self.encodeur.modules))):
            module = self.encodeur.modules[i]
            input_i = self.encodeur.inputs[i]
            module.backward_update_gradient(input_i, delta)
            delta = module.backward_delta(input_i, delta)

    def update_parameters(self, lr):
        self.encodeur.update_parameters(lr)
        self.decodeur.update_parameters(lr)

    def zero_grad(self):
        self.encodeur.zero_grad()
        self.decodeur.zero_grad()



"""
#Test7 (pour la partie classification)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.
x_test = x_test.reshape(-1, 784).astype('float32') / 255.

# One-hot encode les labels
encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_oh = encoder.transform(y_test.reshape(-1, 1))

net1 = Sequentiel()
net1.add_module(Linear(784, 10))
net1.add_module(Softmax())


net2 = Sequentiel()
net2.add_module(Linear(784, 128))
net2.add_module(TanH())
net2.add_module(Linear(128, 10))
net2.add_module(LogSoftmax())

net3 = Sequentiel()
net3.add_module(Linear(784, 128))
net3.add_module(TanH())
net3.add_module(Linear(128, 64))
net3.add_module(TanH())
net3.add_module(Linear(64, 10))
net3.add_module(Softmax())

eps = 0.01
loss_fn = NLLLoss()
optimizer1 = Optim(net1, loss_fn, eps=eps)
optimizer2 = Optim(net2, loss_fn, eps=eps)
optimizer3 = Optim(net3, loss_fn, eps=eps)

batch_size = 32
epochs = 20
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    total_loss = 0
    correct_train = 0

    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i+batch_size]
        batch_y = y_train_oh[i:i+batch_size]
        loss = optimizer2.step(batch_x, batch_y)
        total_loss += loss

        # Accuracy sur batch d'entraînement
        preds = net2.forward(batch_x)
        correct_train += np.sum(np.argmax(preds, axis=1) == np.argmax(batch_y, axis=1))

    avg_train_loss = total_loss / (len(x_train) / batch_size)
    train_acc = correct_train / len(x_train)

    # Test set
    preds_test = net2.forward(x_test)
    test_loss = np.mean(loss_fn.forward(y_test_oh, preds_test))
    test_acc = np.mean(np.argmax(preds_test, axis=1) == y_test)

    # Sauvegarde
    train_losses.append(avg_train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


plt.figure(figsize=(10, 4))
#plt.suptitle(f"Batch size = {batch_size} ", fontsize=14)  # Titre global

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.show()


index = 123  # par exemple
image = x_test[index].reshape(1, -1)  # reshape pour avoir un batch de taille 1
true_label = y_test[index]
# Prédiction
yhat = net2.forward(image)
predicted_label = np.argmax(yhat)
print(f"Vraie étiquette : {true_label} | Prédiction : {predicted_label}")

"""



"""
#TEST8 (pour caculer les differents dimensions latentes et le nombre de couche)

# Charge les données MNIST
(x_train, _), (x_test, _) = mnist.load_data()

# Mise à l’échelle entre 0 et 1 et flatten
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))


nb_dimension_Latente = 64
encoder1 = Sequentiel()
encoder1.add_module(Linear(784, 256))
encoder1.add_module(TanH())
encoder1.add_module(Linear(256, nb_dimension_Latente)) # compression en nb_dimension_Latente dimensions
encoder1.add_module(TanH())

decoder1 = Sequentiel()
decoder1.add_module(Linear(nb_dimension_Latente, 256))
decoder1.add_module(TanH())
decoder1.add_module(Linear(256, 784))
decoder1.add_module(Sigmoid())

autoencodeur1 = AutoEncodeur(encoder1, decoder1)  
optim1 = Optim(autoencodeur1, BCE(), eps=0.01)


encoder2 = Sequentiel()
encoder2.add_module(Linear(784, nb_dimension_Latente)) #compression en 64 dim
encoder2.add_module(TanH())

decoder2 = Sequentiel()
decoder2.add_module(Linear(nb_dimension_Latente, 784))
decoder2.add_module(Sigmoid())

autoencodeur2 = AutoEncodeur(encoder2, decoder2)  
optim2 = Optim(autoencodeur2, BCE(), eps=0.01)


train_losses = []
test_losses = []
for epoch in range(20):
    total_loss_train = 0
    for i in range(0, len(x_train), 100):  # mini-batchs
        batch = x_train[i:i+100]
        loss = optim1.step(batch, batch)
        total_loss_train += loss
    # Moyenne des pertes sur tous les batches
    avg_train_loss = total_loss_train / (len(x_train) / 100)
    train_losses.append(avg_train_loss)

    total_loss_test = 0
    for i in range(0, len(x_test), 100):
        batch = x_test[i:i+100]
        yhat = autoencodeur1.forward(batch)
        loss = BCE().forward(batch, yhat)
        total_loss_test += np.mean(loss)

    avg_test_loss = total_loss_test / (len(x_test) / 100)
    test_losses.append(avg_test_loss)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")


# Sélectionner quelques images test
samples = x_test[110:120]
reconstructed = autoencodeur1.forward(samples)
# Afficher original vs reconstruit
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(samples[i].reshape(28,28), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 10, 10+i+1)
    plt.imshow(reconstructed[i].reshape(28,28), cmap='gray')
    plt.axis('off')

plt.suptitle("Top: Original | Bottom: Reconstruit")
plt.show()


plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Évolution de la perte - Autoencodeur")
plt.legend()
plt.grid(True)
plt.show()
"""

"""
#Test9 (pour analyser les differents fonctions d'activations)
activation_functions = {
    "TanH": TanH,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax, 
    "LogSoftmax"  : LogSoftmax    
}

# Charge les données MNIST
(x_train, _), (x_test, _) = mnist.load_data()

# Mise à l’échelle entre 0 et 1 et flatten
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

def build_autoencoder(latent_dim, activation):
    encoder = Sequentiel()
    encoder.add_module(Linear(784, 256))
    encoder.add_module(activation())
    encoder.add_module(Linear(256, latent_dim))
    encoder.add_module(activation())

    decoder = Sequentiel()
    decoder.add_module(Linear(latent_dim, 256))
    decoder.add_module(activation())
    decoder.add_module(Linear(256, 784))
    decoder.add_module(Sigmoid()) 

    return AutoEncodeur(encoder, decoder)


epochs = 20
batch_size = 100
latent_dim = 64
results = {}

models = {}
optimizers = {}
for name, activation in activation_functions.items():
    model = build_autoencoder(latent_dim, activation)
    models[name] = model
    optimizers[name] = Optim(model, BCE(), eps=0.01)

for name, activation in activation_functions.items():
    print(f"\n--- Entraînement avec {name} ---")
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        # Entraînement
        total_train_loss = 0
        for i in range(0, len(x_train), batch_size):
            batch = x_train[i:i+batch_size]
            loss = optimizers[name].step(batch, batch)
            total_train_loss += loss
        avg_train_loss = total_train_loss / (len(x_train) / batch_size)

        # Évaluation test
        total_test_loss = 0
        for i in range(0, len(x_test), batch_size):
            batch = x_test[i:i+batch_size]
            yhat = models[name].forward(batch)
            total_test_loss += np.mean(BCE().forward(batch, yhat))
        avg_test_loss = total_test_loss / (len(x_test) / batch_size)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch+1:2d} | Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f}")

    results[name] = {"train": train_losses, "test": test_losses}


for name in results:
    plt.plot(results[name]["train"], label=f"{name} - Train")
    plt.plot(results[name]["test"], label=f"{name} - Test")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title(f"fonctions d'activation {name} - Autoencodeur")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

samples = x_test[110:120]  # 10 images
reconstructed_by_activation = {}
for name, activation in activation_functions.items():
    reconstructed_by_activation[name] = models[name].forward(samples)

for i in range(10):
    plt.subplot(5, 10, i + 1)
    plt.imshow(samples[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel("Original", fontsize=10)

# Les reconstructions
for row, (name, reconstructions) in enumerate(reconstructed_by_activation.items(), start=1):
    for i in range(10):
        plt.subplot(5, 10, row * 10 + i + 1)
        plt.imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel(name, fontsize=10)

plt.suptitle("Reconstructions par fonction d’activation", fontsize=14)
plt.tight_layout()
plt.show()
"""


"""#TEST1
# Générer des données simples
np.random.seed(0)
X = np.random.randn(100, 1)  # 100 exemples, 1 feature
true_W = np.array([[2.0]])  # poids vrai
y =  y = X @ true_W + 1 + np.random.randn(100, 1) * 0.1  # Ajout de bruit
 # y = 2x + 1
batch_size = 10
epochs = 100
learning_rate = 0.001

# Créer les objets
model = Linear(1, 1)
loss_fn = MSELoss()

# Boucle d'apprentissage
for epoch in range(epochs):
    model.zero_grad()
    for i in range(0, X.shape[0], batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        yhat = model.forward(batch_X)
        loss = loss_fn.forward(batch_y, yhat)
        delta = loss_fn.backward(batch_y, yhat)

        model.backward_update_gradient(batch_X, delta)
        model.backward_delta(batch_X, delta)  # juste pour tester ici

        model.update_parameters(learning_rate)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {np.mean(loss)}")
print("\nPoids appris :", model._parameters["W"].flatten())
print("Biais appris :", model._parameters["b"].flatten())"""


"""
#TEST2
np.random.seed(0)
X_train = np.random.randn(100, 2)  # 100 exemples, 2 features
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int).reshape(-1, 1)  # Classe 1 si x1 + x2 > 0, sinon 0

batch_size = 10
epochs = 100
learning_rate = 0.01

layer1 = Linear(2, 16)
activation1 = TanH()
layer2 = Linear(16, 1)
activation2 = Sigmoid()
loss_fn = MSELoss()
losses_test2 =[]
for epoch in range(epochs):
    layer1.zero_grad()
    layer2.zero_grad()
    total_loss_train = 0
    for i in range(0, X_train.shape[0], batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        # Forward pass
        out1 = layer1.forward(batch_X)
        act1 = activation1.forward(out1)
        out2 = layer2.forward(act1)
        yhat = activation2.forward(out2)
        loss = loss_fn.forward(batch_y, yhat)
        total_loss_train += loss
        # Backward pass
        delta = loss_fn.backward(batch_y, yhat)
        delta = activation2.backward_delta(out2, delta)
        layer2.backward_update_gradient(act1, delta)
        delta = layer2.backward_delta(act1, delta)
        delta = activation1.backward_delta(out1, delta)
        layer1.backward_update_gradient(batch_X, delta)

    
        layer1.update_parameters(learning_rate)
        layer2.update_parameters(learning_rate)
    losses_test2.append(total_loss_train / (len(X_train) / batch_size))
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {np.mean(loss)}")

print("\nPoids couche 1 :", layer1._parameters["W"])
print("Biais couche 1 :", layer1._parameters["b"])
print("Poids couche 2 :", layer2._parameters["W"])
print("Biais couche 2 :", layer2._parameters["b"])


plt.plot(losses_test2)
plt.xlabel("Epoch")
plt.ylabel("Loss (BCE)")
plt.legend()
plt.grid(True)
plt.show()

"""

"""#TEST3
#Génération des données de classification binaire
np.random.seed(0)
X_train = np.random.randn(100, 2)  # 100 exemples, 2 features
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int).reshape(-1, 1)  # Classe 1 si x1 + x2 > 0

#Hyperparamètres
batch_size = 10
epochs = 100
learning_rate = 0.01

#Création du réseau avec Sequentiel
net = Sequentiel()
net.add_module(Linear(2, 3))     # Entrée 2 → cachée 3
net.add_module(TanH())           # Activation tanh
net.add_module(Linear(3, 1))     # Cachée 3 → sortie 1
net.add_module(Sigmoid())        # Activation sigmoid

#Fonction de perte
loss_fn = MSELoss()

#Boucle d'apprentissage
for epoch in range(epochs):
    net.zero_grad()  # Réinitialise les gradients

    for i in range(0, X_train.shape[0], batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        yhat = net.forward(batch_X)                         # Passe forward
        loss = loss_fn.forward(batch_y, yhat)               # Calcul de la perte
        net.backward(batch_y, yhat, loss_fn)                # Backward global
        net.update_parameters(learning_rate)                # Mise à jour des poids

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {np.mean(loss):.6f}")

# Prédictions sur des données simples pour vérification
X_test = np.array([[1.0, 1.0], [-1.0, -1.0], [0.5, -0.8]])
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int).reshape(-1, 1)

yhat_test = net.forward(X_test)

print("\n Test sur nouvelles données :")
print("Entrées :", X_test)
print("Prédictions (probas) :", yhat_test.flatten())
print("Labels attendus :", y_test.flatten())

#Accuracy sur les données d'entraînement
yhat_train = net.forward(X_train)
predictions = (yhat_train > 0.5).astype(int)
accuracy = np.mean(predictions == y_train)
print(f"\nAccuracy sur l'ensemble d'entraînement : {accuracy:.2%}")"""

"""#TEST4
#Génération des données de classification binaire
np.random.seed(0)
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int).reshape(-1, 1)

#Hyperparamètres
batch_size = 10
epochs = 100
learning_rate = 0.01

#Création du réseau avec Sequentiel
net = Sequentiel()
net.add_module(Linear(2, 3))
net.add_module(TanH())
net.add_module(Linear(3, 1))
net.add_module(Sigmoid())

#Création de la fonction de perte et de l'optimiseur
loss_fn = MSELoss()
optimizer = Optim(net, loss_fn, eps=learning_rate)

#Boucle d'entraînement en utilisant Optim
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        loss = optimizer.step(batch_X, batch_y)  # ici on utilise Optim 

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

#Test sur nouvelles données
X_test = np.array([[1.0, 1.0], [-1.0, -1.0], [0.5, -0.8]])
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int).reshape(-1, 1)
yhat_test = net.forward(X_test)

print("\nTest sur nouvelles données :")
print("Entrées :", X_test)
print("Prédictions (probas) :", yhat_test.flatten())
print("Labels attendus :", y_test.flatten())

#Accuracy sur l'entraînement
yhat_train = net.forward(X_train)
predictions = (yhat_train > 0.5).astype(int)
accuracy = np.mean(predictions == y_train)
print(f"\nAccuracy sur l'ensemble d'entraînement : {accuracy:.2%}")"""

"""#TEST5
#Génération des données de classification binaire
np.random.seed(0)
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int).reshape(-1, 1)

#Création du réseau
net = Sequentiel()
net.add_module(Linear(2, 3))   # Couche cachée
net.add_module(TanH())         # Activation non-linéaire
net.add_module(Linear(3, 1))   # Couche de sortie
net.add_module(Sigmoid())      # Activation binaire

#Définition de la perte et de l'optimiseur
loss_fn = MSELoss()
optimizer = Optim(net, loss_fn, eps=0.01)

#Appel de la fonction SGD (ENTRAÎNEMENT)
SGD(optimizer, X_train, y_train, batch_size=10, epochs=100)

#Test sur nouvelles données
X_test = np.array([[1.0, 1.0], [-1.0, -1.0], [0.5, -0.8]])
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int).reshape(-1, 1)

yhat_test = net.forward(X_test)

print("\n Test sur nouvelles données :")
print("Entrées :", X_test)
print("Prédictions (probas) :", yhat_test.flatten())
print("Labels attendus :", y_test.flatten())

# Accuracy sur l'ensemble d'entraînement
yhat_train = net.forward(X_train)
predictions = (yhat_train > 0.5).astype(int)
accuracy = np.mean(predictions == y_train)
print(f"\nAccuracy sur l'ensemble d'entraînement : {accuracy:.2%}")"""


"""
#TEST6
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#Charger les données
digits = load_digits()
X = digits.data  # 1797 images de 8x8 = 64 features
y = digits.target.reshape(-1, 1)  # labels : 0 à 9

# Normalisation des entrées (entre 0 et 1)
X = X / 16.0

# Encodage one-hot des labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Réseau : Linear -> LogSoftmax
net = Sequentiel()
net.add_module(Linear(64, 32))
net.add_module(TanH())
net.add_module(Linear(32, 10))     # 10 classes
net.add_module(LogSoftmax())       # log(probabilités)

# Fonction de perte : NLLLoss
loss_fn = NLLLoss()

# Optimiseur
optimizer = Optim(net, loss_fn, eps=0.01)

#Entraînement du réseau
SGD(optimizer, X_train, y_train, batch_size=32, epochs=100)

#Évaluation sur le test set
log_probs_test = net.forward(X_test)
probs_test = np.exp(log_probs_test)  # on repasse de log(proba) à proba
predictions = np.argmax(probs_test, axis=1)
true_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == true_labels)
print(f"\nAccuracy sur le test set : {accuracy:.2%}")



"""

