import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Encoder Module (from TS-TCC, Unchanged) ---
# 編碼器與原始版本相同，用於從時間序列中提取特徵。
class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=3):
        super(TSEncoder, self).__init__()
        layers = []
        current_in = input_dims
        for _ in range(depth):
            layers.extend([
                nn.Conv1d(current_in, hidden_dims, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dims),
                nn.ReLU()
            ])
            current_in = hidden_dims
        layers.append(nn.Conv1d(hidden_dims, output_dims, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, seq_len, input_dims]
        x = x.permute(0, 2, 1)  # to [batch, input_dims, seq_len]
        out = self.network(x)
        return out.permute(0, 2, 1)  # to [batch, seq_len, output_dims]


# --- 2. Reconstruction-based TSTCC Model (Memory Removed) ---
# 這是修改後的核心模型。
# 它移除了所有與 MEMTO 記憶體相關的模組。
class Reconstruction_TSTCC(nn.Module):
    def __init__(self, input_dim, latent_dim, n_heads=4, n_layers=2):
        super(Reconstruction_TSTCC, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder from TS-TCC
        self.encoder = TSEncoder(input_dims=input_dim, output_dims=latent_dim)
        
        # --- Modules for TS-TCC Loss (Unchanged) ---
        # TS-TCC 的模組被保留下來，以其對比損失來優化編碼器
        tstcc_encoder_layers = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dim_feedforward=latent_dim * 2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(tstcc_encoder_layers, num_layers=n_layers)
        self.temporal_predictor = nn.Linear(latent_dim, latent_dim)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )

        # --- Decoder (MODIFIED) ---
        # 解碼器被修改，現在只接受潛在向量 q 作為輸入 (latent_dim)，
        # 而不是原始 q 和記憶體的拼接 (latent_dim * 2)。
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),  # 輸入維度從 latent_dim * 2 改為 latent_dim
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, x_primary, x_secondary=None, future_k=5):
        if x_secondary is None:
            x_secondary = x_primary # 用於預測或單一視圖操作

        # --- Feature Extraction ---
        q_primary = self.encoder(x_primary)
        q_secondary = self.encoder(x_secondary)

        # --- TS-TCC Path (for contrastive loss) ---
        c_primary = self.temporal_transformer(q_primary[:, :-future_k, :]).mean(dim=1)
        c_secondary = self.temporal_transformer(q_secondary[:, :-future_k, :]).mean(dim=1)
        
        pred_from_primary = self.temporal_predictor(c_primary).unsqueeze(1).repeat(1, future_k, 1)
        pred_from_secondary = self.temporal_predictor(c_secondary).unsqueeze(1).repeat(1, future_k, 1)
        
        p_primary = self.projector(c_primary)
        p_secondary = self.projector(c_secondary)

        tstcc_outputs = (
            pred_from_primary, q_secondary[:, -future_k:, :],
            pred_from_secondary, q_primary[:, -future_k:, :],
            p_primary, p_secondary
        )

        # --- Reconstruction Path (MODIFIED) ---
        # 移除了所有記憶體查找、更新和拼接的步驟。
        # 直接使用主視圖的特徵 q_primary 進行重建。
        reconstructed_x = self.decoder(q_primary)

        return reconstructed_x, tstcc_outputs

# --- 3. Agent for Training and Prediction (Memory Removed) ---
# 訓練代理也被修改以適應無記憶體模型。
class Agent_NoMemory:
    def __init__(self, input_dim, latent_dim=64, lr=1e-4, future_k=5,
                 l_rec=1.0, l_tstcc_tc=1.0, l_tstcc_cc=0.7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 初始化修改後的無記憶體模型
        self.model = Reconstruction_TSTCC(
            input_dim=input_dim, latent_dim=latent_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.future_k = future_k
        # 移除了與記憶體 entropy loss 相關的 lambda
        self.lambdas = {'rec': l_rec, 'tc': l_tstcc_tc, 'cc': l_tstcc_cc}

    # --- Augmentations (Unchanged) ---
    def _jitter(self, x, sigma=0.1): return x + torch.randn_like(x) * sigma
    def _scale(self, x, sigma=0.1): return x * (torch.randn(x.shape[0], 1, 1, device=self.device) * sigma + 1.0)
    def _permute(self, x, max_segments=5):
        orig_steps = np.arange(x.shape[1]); num_segs = np.random.randint(1, max_segments)
        ret = np.array_split(orig_steps, num_segs); np.random.shuffle(ret)
        return x[:, np.concatenate(ret), :]
    def _get_augmentations(self, x_batch):
        x_weak = self._scale(self._jitter(x_batch))
        x_strong = self._jitter(self._permute(x_batch))
        return x_strong, x_weak

    # --- Loss Calculators (MODIFIED) ---
    # TS-TCC 損失計算保持不變
    def _calculate_tstcc_loss(self, tstcc_outputs, temp=0.2):
        pred_s, z_w_f, pred_w, z_s_f, p_s, p_w = tstcc_outputs
        
        loss_tc = F.mse_loss(pred_s, z_w_f) + F.mse_loss(pred_w, z_s_f)
        
        p_s_norm = F.normalize(p_s, dim=1); p_w_norm = F.normalize(p_w, dim=1)
        sim_matrix = torch.matmul(p_s_norm, p_w_norm.T)
        
        logits = sim_matrix / temp
        labels = torch.arange(logits.shape[0], dtype=torch.long, device=self.device)
        loss_cc = F.cross_entropy(logits, labels)
        
        return self.lambdas['tc'] * loss_tc + self.lambdas['cc'] * loss_cc

    # 損失函數被簡化，只包含重建損失和 TS-TCC 損失。
    def _calculate_combined_loss(self, x_orig, x_rec, tstcc_outputs):
        # Reconstruction loss
        rec_loss = F.mse_loss(x_rec, x_orig)
        
        # TSTCC contrastive loss
        tstcc_loss = self._calculate_tstcc_loss(tstcc_outputs)
        
        total_loss = self.lambdas['rec'] * rec_loss + tstcc_loss
        return total_loss, rec_loss, tstcc_loss

    # --- Training (MODIFIED) ---
    # 訓練流程被簡化為單一階段。
    def train(self, train_loader, epochs=50):
        print("--- Starting Single-Phase Training (Reconstruction + TS-TCC) ---")
        self.model.train()
        for epoch in range(epochs):
            total_loss, total_rec, total_tstcc = 0, 0, 0
            for i, (x_batch,) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                # 弱增強視圖 (weak) 用於重建，強增強視圖 (strong) 用於對比學習
                x_strong, x_weak = self._get_augmentations(x_batch)

                self.optimizer.zero_grad()
                
                # 模型返回重建結果和 TSTCC 的輸出
                x_rec, tstcc_out = self.model(x_weak, x_strong, future_k=self.future_k)
                
                # 計算組合損失
                loss, rec_loss, tstcc_loss = self._calculate_combined_loss(x_weak, x_rec, tstcc_out)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_rec += rec_loss.item()
                total_tstcc += tstcc_loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss/len(train_loader):.4f} "
                  f"(Reconstruction: {total_rec/len(train_loader):.4f}, TS-TCC: {total_tstcc/len(train_loader):.4f})")

    # --- Prediction (MODIFIED) ---
    # 預測函數現在只依賴重建誤差 (ISD) 作為異常分數。
    def predict(self, test_loader):
        self.model.eval()
        anomaly_scores = []
        with torch.no_grad():
            for (x_batch,) in test_loader:
                x_batch = x_batch.to(self.device)
                
                # 從模型獲取重建輸出
                reconstructed_x, _ = self.model(x_batch)
                
                # 計算 ISD (Input Space Deviation)，即重建誤差
                isd = torch.pow(x_batch - reconstructed_x, 2).sum(dim=-1)
                
                # (REMOVED) 不再計算 LSD (Latent Space Deviation)
                
                # 異常分數就是 ISD
                score = isd
                anomaly_scores.append(score.cpu().numpy())
        
        return np.concatenate(anomaly_scores, axis=0)