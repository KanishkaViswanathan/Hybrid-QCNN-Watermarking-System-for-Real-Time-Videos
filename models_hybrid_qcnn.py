import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

# Quantum block
N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)

def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

qnode = qml.QNode(circuit, dev, interface="torch")
weight_shapes = {"weights": (2, N_QUBITS)}
QuantumLayer = qml.qnn.TorchLayer(qnode, weight_shapes)

class EncoderHybridQCNN(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

        self.conv1 = conv_block(4, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.to_q = nn.Linear(64, N_QUBITS)
        self.q_layer = QuantumLayer
        self.from_q = nn.Linear(N_QUBITS, 64)

        # Residual head STILL CNN (important!)
        self.residual_head = nn.Sequential(
            conv_block(64, 32),
            conv_block(32, 32),
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, frame, wm_logo):
        wm_up = F.interpolate(wm_logo, size=frame.shape[-2:], mode="bilinear")
        x = torch.cat([frame, wm_up], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        vec = self.pool(x).squeeze(-1).squeeze(-1)

        q_in = torch.tanh(self.to_q(vec))
        q_out = self.q_layer(q_in)

        refined = self.from_q(q_out)
        refined_map = refined.unsqueeze(-1).unsqueeze(-1).expand_as(x)

        # Fusion (THIS IS THE KEY CHANGE)
        fused = x + refined_map

        residual = self.residual_head(fused)

        watermarked = frame + self.alpha * torch.tanh(residual)
        watermarked = torch.clamp(watermarked, 0, 1)

        return watermarked, residual


class DecoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            conv_block(32, 32),
        )
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.net(x)
        x = self.out(x)
        x = F.interpolate(x, size=(64, 64))
        return torch.sigmoid(x)


class WatermarkHybridQCNN(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.encoder = EncoderHybridQCNN(alpha)
        self.decoder = DecoderCNN()

    def forward(self, frame, wm_logo):
        watermarked, residual = self.encoder(frame, wm_logo)
        wm_hat = self.decoder(watermarked)
        return watermarked, wm_hat, residual

if __name__ == "__main__":
    B = 2
    frame = torch.rand(B, 3, 256, 256)
    wm = torch.rand(B, 1, 64, 64)

    model = WatermarkHybridQCNN(alpha=0.05)
    watermarked, wm_hat, residual = model(frame, wm)

    print("watermarked:", watermarked.shape)
    print("wm_hat:", wm_hat.shape)
    print("residual:", residual.shape)