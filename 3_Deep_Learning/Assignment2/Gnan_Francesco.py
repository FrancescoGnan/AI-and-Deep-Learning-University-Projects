'''
Assignment 2
Student: NAME SURNAME
'''
# *** Packges ***
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# *** Functions ***

if __name__ == "__main__":
    # Write your code here
    print('Hello world!')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


'''
DON'T MODIFY THE SEED!
'''
# Set the seed for reproducibility
manual_seed = 42
torch.manual_seed(manual_seed)

'''
Assignment

'''
# Do all the assignment here

# ============================== 1.1.1 (load and inspect the data)

# reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# Load and normalize the CIFAR10 training and test datasets using torchvision

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].


# train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
# test
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# inspection: we observe one image per class

class_images = {class_name: None for class_name in classes}
for image, label in trainset:
    class_name = classes[label]
    if class_images[class_name] is None:
        class_images[class_name] = image

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i, (class_name, image) in enumerate(class_images.items()):
    axes[i].imshow(image)
    axes[i].set_title(class_name)
    axes[i].axis('off')
plt.savefig("images.pdf")
plt.show()

# inspection: histogram of the distribution of the images of the training and test set
# how many cats, dogs,...

def create_histogram(set):
  histo = {}
  for _, label in set:
      class_ = classes[label]
      if class_ in histo:
        histo[class_] += 1
      else:
        histo[class_] = 1

  return histo

def plot_histograms(hist1, hist2, tit1, tit2):
  labels1 = list( hist1.keys() )
  counts1 = list( hist1.values() )
  labels2 = list( hist2.keys() )
  counts2 = list( hist2.values() )

  plt.title( "Distribution of the images of the training and test set" )
  plt.bar(labels1, counts1, color='b', label=tit1)
  plt.bar(labels2, counts2, color='r', label=tit2)
  plt.xlabel('Classes')
  plt.ylabel('Counts')
  plt.legend(frameon=True, framealpha=1)
  plt.savefig("histo.pdf")
  plt.show()


histo_train = create_histogram(trainset)
histo_test = create_histogram(testset)

plot_histograms(histo_train, histo_test, 'Train', 'Test')

# ============================== 1.1.2 (Transform the data into tensors)

# The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.
# To work in PyTorch we want each entry to be a torch.Tensor of shape (32, 32, 3)

# i)
dataiter = iter(trainset)
sample = next(dataiter)
print('\nBefore transformation into tensors:')
print(type(sample))
for element in sample:
    print(type(element))
# Before the tranformation, each element of the dataset is a tuple.
# The tuple is composed by a PIL.Image.Image, which is the image itself, and
# a int value, which is the label information
# ii)
to_tensor = transforms.ToTensor()
trainset = [ (to_tensor(image), label) for image, label in trainset]
testset = [ (to_tensor(image), label) for image, label in testset]
# So the entries are not in the correct type for the DL framework in PyTorch.
# This is why during the download we convert the dataset elements into
# tensors and we normalize them using torchvision.transforms.
# iii)
dataiter = iter(trainset)
images, labels = next( dataiter)
image_dimensions = images.shape
print("\nDimensions of one image after transformation into torch.Tensor:", image_dimensions)
# We observe that after the transformation the dimension of each image is
# torch.Size([3, 32, 32]), where 3 is the number of color channels,
# 32x32 is the image size in pixels.

# ============================== 1.1.3 (Normalize the data)

# In order to work with features having mean 0 and standard deviation equal to 1,
# we use torchvision.transforms.Normalize
to_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
trainset = [ (to_norm(image), label) for image, label in trainset]
testset = [ (to_norm(image), label) for image, label in testset]

# ============================== 1.1.4 (Validation dataset)
# reference: https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4

lengths = [int(len(trainset)*0.8), int(len(trainset)*0.2)]
training_dataset, validation_dataset = torch.utils.data.random_split(trainset, lengths)


# ============================== 1.2 (Model)
# We shuffle and batch the training dataset using DataLoader
# because we want to expose our model to many examples in different
# orders during training. Then we set shuffle=True for the training set and
# shuffle=False for the validation and the test ones. This ensures that
# the model is exposed to diverse and random samples during training while
# maintaining a consistent order for evaluation on the validation and test datasets.
# We evaluate the model's performances on the validation and test datasets
# without updating the model parameters. This is why we don't need
# to shuffle the order of the examples.

batch_size = 50

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

class ConvNet(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
    
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=46, kernel_size=2, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=46, out_channels=46, kernel_size=2, stride=1, padding=0)

    self.conv5 = nn.Conv2d(in_channels=46, out_channels=64, kernel_size=2, stride=1, padding=0)
    self.conv6 = nn.Conv2d(in_channels=64, out_channels=72, kernel_size=2, stride=1, padding=0)

    self.pool = nn.MaxPool2d(2, stride=2, padding=0)

    fc_size = 2 * 2 * 72
    self.fc1 = nn.Linear(fc_size, 120)
    self.fc2 = nn.Linear(120, 20)
    self.fc3 = nn.Linear(20, 10)


  def forward(self, x):
    x = (F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    
    x = (F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))

    x = (F.relu(self.conv5(x)))
    x = self.pool(F.relu(self.conv6(x)))

    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


"""
input image size is (3, 32, 32)
output = (input - kernel + 2*padding)/stride + 1

conv1 = (32 - 3) + 1 = 30
conv2 = (30 - 3) + 1 = 28
pool1 = (28 - 2)/2 + 1 = 14
conv3 = (14 - 2) + 1 = 13
conv4 = (13 - 2) + 1 = 12
pool2 = (12 - 2)/2 + 1 = 6
conv5 = (6 -  2) + 1 = 5
conv6 = (5 -  2) + 1 = 4
pool3 = (4 - 2)/2 + 1 = 2
"""
net = ConvNet()
print('Model parameters:\t',sum(p.numel() for p in net.parameters() if p.requires_grad))

# ============================== 1.3.1 (Training)

net = ConvNet()
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
mom = 0.9
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=mom)
n_epochs = 50
n_steps = batch_size


# -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:")
print(DEVICE)

net = net.to(DEVICE)
# -----------------


training_loss = []
validation_loss = []
train_accuracy = []
val_accuracy = []

train_loss_epoch = []
val_loss_epoch = []
train_accu_epoch = []
val_accu_epoch = []

for epoch in range(n_epochs):
  j = 0
  # ------------- training -------------
  for i, (images,labels) in enumerate(train_loader,0):
      # convert labels to binary (0 or 1) based on whether they are equal to 3 or not.
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      net.train()
      optimizer.zero_grad()
      # Forward pass
      outputs = net(images)
      loss_train = loss_fn(outputs, labels)
      # Backward and optimize
      loss_train.backward()
      optimizer.step()
        
      _, predicted = torch.max(outputs, 1)
      correct = (predicted == labels).sum().item()
      t_accuracy = correct / len(labels)

      j += 1
      if j == 4:
          training_loss.append(loss_train.item())
          train_accuracy.append(t_accuracy * 100)
          j = 0
  # ------------- validation -------------
  for i, (images,labels) in enumerate(val_loader):
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)
      
      net.eval()
      with torch.no_grad():
        hat_labels = net(images)
        # Compute the loss
        loss_eval = loss_fn(hat_labels,labels)
        
        _, predicted = torch.max(hat_labels, 1)
        correct = (predicted == labels).sum().item()
        v_accuracy = correct / len(labels)
            
        val_accuracy.append(v_accuracy * 100)
        validation_loss.append(loss_eval.item())  
          
  train_loss_epoch.append(loss_train.item())
  train_accu_epoch.append(t_accuracy * 100)        
  val_accu_epoch.append(v_accuracy * 100)
  val_loss_epoch.append(loss_eval.item()) 
  print("Training:\t Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch + 1, n_epochs, i, len(train_loader), loss_train.item(), t_accuracy* 100))
  print("Valiation:\t Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%\n".format(epoch + 1, n_epochs, i, len(val_loader), loss_eval.item(), v_accuracy* 100))
  print('--------')

# ============================== 1.3.2

net.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = (correct_test / total_test) * 100
print(f'Accuracy on the test dataset: {test_accuracy:.2f}%')

# ----------------------------
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(DEVICE)
labels = labels.to(DEVICE)
outputs = net(images)
probs = F.softmax(outputs, dim=1)
predicted_indices = torch.argmax(probs, dim=1)
print('Predicted classes: ', predicted_indices)
count = 0
for i in range(len(images)):
    true_class = classes[labels[i].item()]
    predicted_class = classes[predicted_indices[i].item()]
    if true_class == predicted_class:
      count += 1
    print(f'True Class: {true_class:5s}, Predicted Class: {predicted_class:5s}')
print(f'It got {count}/{len(images)} classes')

# ============================== 1.3.3 (Save model parameters)

model_name = "FRANCESCO_GNAN_1.pt"
torch.save(net.state_dict(), model_name)

# net.load_state_dict(torch.load(model_name)) to load them later

# ============================== 1.3.4 (Plots)

x = np.arange(n_epochs)
plt.figure()
plt.title("Model 1: Training loss v.s. Validation loss (Epochs)")
plt.plot(x, train_loss_epoch,color='b',label='Training')
plt.plot(x, val_loss_epoch,color='orange',label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model1_ep_loss.pdf')
plt.show()

x = np.arange(len(validation_loss))
plt.figure()
plt.title("Model 1: Training loss v.s. Validation loss")
plt.plot(x, training_loss,color='b',label='Training')
plt.plot(x, validation_loss,color='orange',label='Validation')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model1_it_loss.pdf')
plt.show()

x = np.arange(n_epochs)
plt.figure()
plt.title("Model 1: Training accuracy v.s. Validation accuracy (Epochs)")
plt.plot(x, train_accu_epoch,color='b',label='Training')
plt.plot(x, val_accu_epoch,color='orange',label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model1_ep_accu.pdf')
plt.show()

x = np.arange(len(val_accuracy))
plt.figure()
plt.title("Model 1: Training accuracy v.s. Validation accuracy")
plt.plot(x, train_accuracy,color='b',label='Training')
plt.plot(x, val_accuracy,color='orange',label='Validation')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model1_it_accu.pdf')
plt.show()

# ============================== 1.3.5 (Model 2)
    
'''
cov1 = 32 - 3 + 1 = 30
conv2 = 30 - 3 + 1 = 28
pool1 = 28 - 2)/2 + 1 = 14
conv3 = 14 - 2 + 1 = 13
conv4 = 13 - 2 + 1 = 12
pool2 = 12 - 2)/2 + 1 = 6
conv5 = 6 - 2 + 1 = 5
conv5 = 5 - 2 + 1 = 4
pool3 = 4 - 2)/2 + 1 = 2 
conv6 = 2 - 1 + 1 = 2
conv7 = 2 - 1 + 1 = 2

'''

class ConvNet2(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0)

    self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)

    self.conv5 = nn.Conv2d(in_channels=64, out_channels=120, kernel_size=2, stride=1, padding=0)
    self.conv6 = nn.Conv2d(in_channels=120, out_channels=120, kernel_size=2, stride=1, padding=0)

    self.conv7 = nn.Conv2d(in_channels=120, out_channels=220, kernel_size=1, stride=1, padding=0)

    self.pool = nn.MaxPool2d(2, stride=2, padding=0)

    fc_size = 2 * 2 * 220
    self.fc1 = nn.Linear(fc_size, 100)
    self.fc2 = nn.Linear(100, 60)
    self.fc3 = nn.Linear(60, 10)

    self.dropout1 = nn.Dropout(0.2)
    self.dropout2 = nn.Dropout(0.3)
    self.dropout3 = nn.Dropout(0.5)

    self.batchnorm1 = nn.BatchNorm2d(48)
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.batchnorm3 = nn.BatchNorm2d(120)
    self.batchnorm4 = nn.BatchNorm2d(220)

    self.batchnorm_fc1 = nn.BatchNorm1d(100)
    self.batchnorm_fc2 = nn.BatchNorm1d(60)



  def forward(self, x):
    
    x = F.gelu(self.conv1(x))
    x = self.pool(F.gelu(self.batchnorm1(self.conv2(x))))
    x = self.dropout1(x)
    
    x = F.gelu(self.conv3(x))
    x = self.pool(F.gelu(self.batchnorm2(self.conv4(x))))
    x = self.dropout2(x)
    
    x = F.gelu(self.conv5(x))
    x = self.pool(F.gelu(self.batchnorm3(self.conv6(x))))
    x = self.dropout3(x)
    
    x = F.gelu(self.batchnorm4(self.conv7(x)))


    x = torch.flatten(x, 1)
    x = F.gelu(self.batchnorm_fc1(self.fc1(x)))
    x = self.dropout1(x)
    x = F.gelu(self.batchnorm_fc2(self.fc2(x)))

    x = self.fc3(x)
    return x

net = ConvNet2()

print('Model parameters:\t',sum(p.numel() for p in net.parameters() if p.requires_grad))

# --------------- Training

net2 = ConvNet2()
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(net2.parameters(), lr=learning_rate)
n_epochs = 50
n_steps = 500


# -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:")
print(DEVICE)

net2 = net2.to(DEVICE)
# -----------------


training_loss = []
validation_loss = []
train_accuracy = []
val_accuracy = []

train_loss_epoch = []
val_loss_epoch = []
train_accu_epoch = []
val_accu_epoch = []

for epoch in range(n_epochs):
  j = 0
  # ------------- training -------------
  for i, (images,labels) in enumerate(train_loader,0):
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      net2.train()
      optimizer.zero_grad()
      # Forward pass
      outputs = net2(images)
      loss_train = loss_fn(outputs, labels)
      # Backward and optimize
      loss_train.backward()
      optimizer.step()
        
      _, predicted = torch.max(outputs, 1)
      correct = (predicted == labels).sum().item()
      t_accuracy = correct / len(labels)

      j += 1
      if j == 4:
          training_loss.append(loss_train.item())
          train_accuracy.append(t_accuracy * 100)
          j = 0
  # ------------- validation -------------
  for i, (images,labels) in enumerate(val_loader):
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)
      
      net2.eval()
      with torch.no_grad():
        hat_labels = net2(images)
        # Compute the loss
        loss_eval = loss_fn(hat_labels,labels)
        
        _, predicted = torch.max(hat_labels, 1)
        correct = (predicted == labels).sum().item()
        v_accuracy = correct / len(labels)
            
        val_accuracy.append(v_accuracy * 100)
        validation_loss.append(loss_eval.item())  
          
  train_loss_epoch.append(loss_train.item())
  train_accu_epoch.append(t_accuracy * 100)        
  val_accu_epoch.append(v_accuracy * 100)
  val_loss_epoch.append(loss_eval.item()) 
  print("Training:\t Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch + 1, n_epochs, i, len(train_loader), loss_train.item(), t_accuracy* 100))
  print("Valiation:\t Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%\n".format(epoch + 1, n_epochs, i, len(val_loader), loss_eval.item(), v_accuracy* 100))
  print('--------')

  #-------------- test accuracy

net2.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = net2(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = (correct_test / total_test) * 100
print(f'Accuracy on the test dataset: {test_accuracy:.2f}%')

# ============================== 1.3.6 (Save)

model_name = "FRANCESCO_GNAN_2.pt"
torch.save(net2.state_dict(), model_name)

# net2.load_state_dict(torch.load(model_name)) to load them later

#-------------- Plots

x = np.arange(n_epochs)
plt.figure()
plt.title("Model 2: Training loss v.s. Validation loss (Epochs)")
plt.plot(x, train_loss_epoch,color='b',label='Training')
plt.plot(x, val_loss_epoch,color='orange',label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model2_ep_loss.pdf')
plt.show()

x = np.arange(len(validation_loss))
plt.figure()
plt.title("Model 2: Training loss v.s. Validation loss")
plt.plot(x, training_loss,color='b',label='Training')
plt.plot(x, validation_loss,color='orange',label='Validation')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model2_it_loss.pdf')
plt.show()

x = np.arange(n_epochs)
plt.figure()
plt.title("Model 2: Training accuracy v.s. Validation accuracy (Epochs)")
plt.plot(x, train_accu_epoch,color='b',label='Training')
plt.plot(x, val_accu_epoch,color='orange',label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model2_ep_accu.pdf')
plt.show()

x = np.arange(len(val_accuracy))
plt.figure()
plt.title("Model 2: Training accuracy v.s. Validation accuracy")
plt.plot(x, train_accuracy,color='b',label='Training')
plt.plot(x, val_accuracy,color='orange',label='Validation')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model2_it_accu.pdf')
plt.show()

#-------------- How many classes does it predict?
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(DEVICE)
labels = labels.to(DEVICE)
outputs = net2(images)
probs = F.softmax(outputs, dim=1)
predicted_indices = torch.argmax(probs, dim=1)
count = 0
for i in range(len(images)):
    true_class = classes[labels[i].item()]
    predicted_class = classes[predicted_indices[i].item()]
    if true_class == predicted_class:
      count += 1
    print(f'True Class: {true_class:5s}, Predicted Class: {predicted_class:5s}')
print(f'It got {count}/{len(images)} classes')


#======================= BONUS


'''
Code for bonus question
'''
for seed in range(10):
    torch.manual_seed(seed)
    # Train the models here
    # ============================== 1.3.1 (Training)
    print(f'seed: {seed}')
    
    net2 = ConvNet2()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(net2.parameters(), lr=learning_rate)
    n_epochs = 10
    n_steps = 500


    # -----------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:")
    print(DEVICE)

    net2 = net2.to(DEVICE)
    # -----------------
    
    for epoch in range(n_epochs):
      # ------------- training -------------
      for i, (images,labels) in enumerate(train_loader,0):
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)

          net2.train()
          optimizer.zero_grad()
          # Forward pass
          outputs = net2(images)
          loss_train = loss_fn(outputs, labels)
          # Backward and optimize
          loss_train.backward()
          optimizer.step()
        
   
      # ------------- validation -------------
      for i, (images,labels) in enumerate(val_loader):
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)
      
          net2.eval()
          with torch.no_grad():
            hat_labels = net2(images)
            # Compute the loss
            loss_eval = loss_fn(hat_labels,labels)

            
    net2.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net2(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = (correct_test / total_test) * 100
    print(f'Accuracy on the test dataset: {test_accuracy:.2f}%')

      
    
