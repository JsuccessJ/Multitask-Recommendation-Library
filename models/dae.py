import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron


class DenoiseAutoEncoder(torch.nn.Module):
    """
    Denoising AutoEncoder 구현
    
    노이즈가 추가된 입력을 받아 원본 데이터를 복원하도록 학습하는 모델입니다.
    """
    def __init__(self, input_dim, bottom_mlp_dims, dropout, noise_ratio=0.2):
        super().__init__()
        self.noise_ratio = noise_ratio
        
        # 인코더 레이어 구성 - 튜플을 리스트로 변환
        encoder_dims = [input_dim] + list(bottom_mlp_dims)
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(torch.nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            encoder_layers.append(torch.nn.BatchNorm1d(encoder_dims[i+1]))
            encoder_layers.append(torch.nn.ReLU())
            encoder_layers.append(torch.nn.Dropout(dropout))
        self.encoder = torch.nn.Sequential(*encoder_layers)
        
        # 디코더 레이어 구성 - 튜플을 리스트로 변환
        decoder_dims = list(bottom_mlp_dims)[::-1] + [input_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_dims[i+1]))
                decoder_layers.append(torch.nn.ReLU())
                decoder_layers.append(torch.nn.Dropout(dropout))
        self.decoder = torch.nn.Sequential(*decoder_layers)
        
    def add_noise(self, x):
        """입력에 노이즈 추가"""
        noise_mask = torch.bernoulli(torch.ones_like(x) * self.noise_ratio)
        return x * (1 - noise_mask)
    
    def forward(self, x, add_noise=True):
        """
        :param x: 입력 텐서
        :param add_noise: 노이즈 추가 여부 (학습 시 True, 추론 시 False)
        """
        if add_noise and self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
            
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        
        return encoded  # expert로 사용할 때는 인코딩된 표현만 반환


class OMoEDAEModel(torch.nn.Module):
    """
    DenoiseAutoEncoder를 expert로 사용하는 One-gate MoE 모델 구현
    
    Reference:
        Jacobs, Robert A., et al. "Adaptive mixtures of local experts." Neural computation 3.1 (1991): 79-87.
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, 
                 task_num, expert_num, dropout, noise_ratio=0.2):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num

        # DAE expert 레이어 구성
        self.expert = torch.nn.ModuleList([
            DenoiseAutoEncoder(self.embed_output_dim, bottom_mlp_dims, dropout, noise_ratio) 
            for i in range(expert_num)
        ])
        
        # 각 task에 대한 tower 레이어
        self.tower = torch.nn.ModuleList([
            MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) 
            for i in range(task_num)
        ])
        
        # Gate 네트워크
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(self.embed_output_dim, expert_num), 
            torch.nn.Softmax(dim=1)
        )

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        
        # Gate 값 계산
        gate_value = self.gate(emb).unsqueeze(1)
        
        # 각 expert의 출력 계산 (학습 중에는 노이즈 추가, 추론 시에는 노이즈 미추가)
        fea = torch.cat([self.expert[i](emb, add_noise=self.training).unsqueeze(1) 
                         for i in range(self.expert_num)], dim=1)
        
        # Gate 기반 expert 출력 결합
        fea = torch.bmm(gate_value, fea).squeeze(1)
        
        # 각 task에 대한 예측
        results = [torch.sigmoid(self.tower[i](fea).squeeze(1)) for i in range(self.task_num)]
        return results
