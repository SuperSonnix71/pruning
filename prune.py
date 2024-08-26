import torch
import torch.nn as nn
import logging
from torchsummary import summary


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    logging.info(f"Using GPU: {gpu_names} (Total: {num_gpus})")
else:
    device = torch.device("cpu")
    logging.info("No GPU detected. Using CPU.")

class SimpleNet(nn.Module):
    def __init__(self):
        """Initializes the SimpleNet model with four linear layers."""
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 10)
        logging.info("Original SimpleNet initialized.")

    def forward(self, x):
        """Forward pass of the model."""
        try:
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
            x = self.layer4(x)
        except Exception as e:
            logging.error("Error during forward pass: %s", e)
            raise
        return x

class DepthPrunedNet(nn.Module):
    def __init__(self):
        """Initializes the DepthPrunedNet model by removing one layer from SimpleNet."""
        super(DepthPrunedNet, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 10)  # <--------------------Removed layer3
        logging.info("DepthPrunedNet initialized with one less layer.")

    def forward(self, x):
        """Forward pass of the depth-pruned model."""
        try:
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = self.layer4(x)
        except Exception as e:
            logging.error("Error during forward pass in DepthPrunedNet: %s", e)
            raise
        return x

class WidthPrunedNet(nn.Module):
    def __init__(self):
        """Initializes the WidthPrunedNet model by reducing the number of neurons in one layer."""
        super(WidthPrunedNet, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 64)  # <--------------- Reduced neurons from 128 to 64
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 10)
        logging.info("WidthPrunedNet initialized with reduced neurons in layer2.")

    def forward(self, x):
        """Forward pass of the width-pruned model."""
        try:
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
            x = self.layer4(x)
        except Exception as e:
            logging.error("Error during forward pass in WidthPrunedNet: %s", e)
            raise
        return x

def initialize_and_summarize_models():
    """Initializes and summarizes all three models."""
    try:
        model = SimpleNet().to(device)
        depth_pruned_model = DepthPrunedNet().to(device)
        width_pruned_model = WidthPrunedNet().to(device)

        logging.info("Model Summary for SimpleNet:")
        summary(model, (1, 784))

        logging.info("Model Summary for DepthPrunedNet:")
        summary(depth_pruned_model, (1, 784))

        logging.info("Model Summary for WidthPrunedNet:")
        summary(width_pruned_model, (1, 784))

    except Exception as e:
        logging.error("Error initializing or summarizing models: %s", e)
        raise

if __name__ == "__main__":
    logging.info("Starting model initialization and summary process.")
    initialize_and_summarize_models()
    logging.info("Completed model initialization and summary process.")
