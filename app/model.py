"""Integrity Theory quick-scan model (v5.1 weights).

Everything between the "VERBATIM" markers is copied unchanged from predict_v4.py in
https://github.com/kausikragavs/Integrity-theory (only CRLF line endings became LF).
Do not edit those blocks: the trained weights in weights/integrity_model_v5.1.pth
only load into this exact architecture.

Below the copied code, preprocess_upload() repeats the steps of preprocess_image()
one for one, but works on an uploaded file held in memory instead of a path on disk,
and also hands back the intermediate values the explanation views need.
checks/acceptance.py confirms both functions produce bit-identical tensors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


# ===== VERBATIM from predict_v4.py: model classes =====
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) 
        out += self.shortcut(x) 
        return F.relu(out)

class IntegrityResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.stage2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.stage3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_avg_pool(x)
        return torch.flatten(x, 1)

class FrequencyNetwork(nn.Module):
    def __init__(self, input_size=1024): 
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.relu(self.fc2(x))

class IntegrityFusionModelV4(nn.Module):
    def __init__(self, num_classes=2): 
        super().__init__()
        self.spatial_branch = IntegrityResNet()
        self.freq_branch = FrequencyNetwork(input_size=1024) 
        fused_dim = 128 + 64 + 2
        
        self.fusion_fc1 = nn.Linear(fused_dim, 128)
        self.dropout = nn.Dropout(0.4)
        self.fusion_fc2 = nn.Linear(128, num_classes)

    def forward(self, images, freq_tensors, metadata_flags):
        spatial_features = self.spatial_branch(images)
        freq_features = self.freq_branch(freq_tensors)
        fused_vector = torch.cat((spatial_features, freq_features, metadata_flags), dim=1)
        x = F.relu(self.fusion_fc1(fused_vector))
        x = self.dropout(x)
        return self.fusion_fc2(x)


# ===== VERBATIM from predict_v4.py: preprocess_image() (reference; used by checks/acceptance.py) =====
#inference logic
def preprocess_image(img_path):
    spatial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None, None, None, None
        
    image_tensor = spatial_transform(img)
    
    has_exif = 1.0 if img.getexif() else 0.0
    has_c2pa = 0.0
    try:
        with open(img_path, 'rb') as f:
            file_bytes = f.read()
            if b'JUMBF' in file_bytes or b'c2pa' in file_bytes:
                has_c2pa = 1.0
    except Exception:
        pass
    metadata_flags = torch.tensor([has_exif, has_c2pa], dtype=torch.float32)
    
    # Standardized 256x256 Grid FFT Processing
    fft_resize = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    gray_tensor = fft_resize(img).squeeze(0)
    
    fft_result = torch.fft.fft2(gray_tensor)
    fft_shifted = torch.fft.fftshift(fft_result)
    magnitude_spectrum = torch.abs(fft_shifted)
    
    center_y, center_x = 128, 128 
    cropped_fft = magnitude_spectrum[center_y-16:center_y+16, center_x-16:center_x+16]
    cropped_fft = torch.log(1 + cropped_fft)
    freq_tensor = cropped_fft.flatten() 
    
    return image_tensor.unsqueeze(0), freq_tensor.unsqueeze(0), metadata_flags.unsqueeze(0), has_c2pa

# ===== END VERBATIM =====


# --- Same steps as preprocess_image(), for an upload held in memory -------------------
# The transforms are built exactly as in preprocess_image(); they are created once here
# instead of on every call, which does not change their output.
SPATIAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
FFT_RESIZE = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
CLASS_NAMES = ["Real", "AI-generated (Fake)"]  # class 0 = Real, class 1 = Fake


def preprocess_upload(img, file_bytes):
    """Mirror of preprocess_image().

    img        -- the uploaded image after Image.open(...).convert('RGB'), exactly as
                  preprocess_image() does it (EXIF is read from this converted image).
    file_bytes -- the raw uploaded bytes (what preprocess_image() reads with open(..., 'rb')).

    Returns the three model inputs plus the full 256x256 magnitude spectrum for the
    frequency view.
    """
    image_tensor = SPATIAL_TRANSFORM(img)

    has_exif = 1.0 if img.getexif() else 0.0
    has_c2pa = 0.0
    if b'JUMBF' in file_bytes or b'c2pa' in file_bytes:
        has_c2pa = 1.0
    metadata_flags = torch.tensor([has_exif, has_c2pa], dtype=torch.float32)

    gray_tensor = FFT_RESIZE(img).squeeze(0)

    fft_result = torch.fft.fft2(gray_tensor)
    fft_shifted = torch.fft.fftshift(fft_result)
    magnitude_spectrum = torch.abs(fft_shifted)

    center_y, center_x = 128, 128
    cropped_fft = magnitude_spectrum[center_y-16:center_y+16, center_x-16:center_x+16]
    cropped_fft = torch.log(1 + cropped_fft)
    freq_tensor = cropped_fft.flatten()

    return {
        "image": image_tensor.unsqueeze(0),        # [1, 3, 224, 224]
        "freq": freq_tensor.unsqueeze(0),          # [1, 1024]
        "meta": metadata_flags.unsqueeze(0),       # [1, 2] = [has_exif, has_c2pa]
        "has_exif": has_exif,
        "has_c2pa": has_c2pa,
        "magnitude_spectrum": magnitude_spectrum,  # [256, 256], before the crop and log
    }


def load_model(weights_path):
    """Load the v5.1 weights the way the brief specifies: CPU, then eval()."""
    model = IntegrityFusionModelV4(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model
