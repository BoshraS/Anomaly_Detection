import torch
import ConvAE_256res

data_shape  = [3, 256]
latent_dim  = 64
n_kernels   = 32
kernel_size = 4

model = ConvAE_256res.ConvAutoEncoder(x_dim=data_shape, latent_dim=latent_dim, n_kernels=n_kernels, kernel_size=kernel_size)

# Load the trained model state dict
model.load_state_dict(torch.load("./models/AE45.pth"))

# Convert the model to evaluation mode
model.eval()

# Convert the model to TorchScript and save
scripted_model = torch.jit.script(model)
scripted_model.save("conv_autoencoder_scripted.pt")