#!/usr/bin/env python
"""
EXPERIMENTO 01: 5-fold Stratified Cross-Validation com R3D-18 pré-treinado
Baseline principal para artigo de doutorado - Resultados novos e robustos

REQUISITOS ATENDIDOS:
1. ✅ 5-fold Stratified CV (protocolo principal)
2. ✅ R3D-18 pretrained (Kinetics-400)
3. ✅ BCEWithLogitsLoss com pos_weight automático (padrão científico)
4. ✅ AMP ativado (RTX 5050 8GB otimizado)
5. ✅ Batch_size=16 (com AMP)
6. ✅ 50 épocas máximo, early stopping patience=15
7. ✅ Seed fixa=42 para reprodutibilidade
8. ✅ AUC consistente (y_true/y_prob salvos)
9. ✅ Oversampling apenas no treino
10.✅ Log completo do sistema

Autor: [Seu Nome]
Data: 2026
Instituição: CEFET/RJ
"""
import os
import sys
import json
import time
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import platform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F

from torchvision.models.video import r3d_18, R3D_18_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, 
    precision_recall_curve, auc, confusion_matrix,
    roc_curve
)
from imblearn.over_sampling import RandomOverSampler

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES PRINCIPAIS - AJUSTADAS PARA RTX 5050 8GB
# ============================================================================
class Config:
    # Paths (Windows 11)
    DATA_DIR = Path(r"C:\dataset\data")
    OUTPUT_DIR = Path(r"C:\dataset\runs")
    EXP_NAME = "exp01_cv_pretrained"
    
    # Arquivos de dados
    NORMAL_FILE = "normal-3DESS-128-64.npy"
    ABNORMAL_FILE = "abnormal-3DESS-128-64.npy"
    
    # Hiperparâmetros otimizados para RTX 5050 8GB
    BATCH_SIZE = 16                    # AMP permite batch maior
    NUM_EPOCHS = 50                    # Como no artigo original
    EARLY_STOP_PATIENCE = 15           # Mais agressivo que artigo
    
    # Otimização (valores padrão científicos)
    LEARNING_RATE = 1e-4               # Adam padrão
    WEIGHT_DECAY = 1e-5                # Regularização leve
    
    # Modelo
    PRETRAINED = True                  # Kinetics-400
    LOSS_TYPE = "bce"                  # BCEWithLogitsLoss (padrão)
    POS_WEIGHT_AUTO = True             # Calcular automaticamente
    
    # Augmentation
    AUGMENTATION = True
    
    # GPU Performance
    AMP = True                         # Mixed Precision (CRÍTICO para batch 16)
    NUM_WORKERS = 0                    # Windows safe
    PIN_MEMORY = True
    
    # Validação Cruzada
    N_SPLITS = 5                       # 5-fold Stratified CV
    SEED = 42                          # Reprodutibilidade total
    
    # Oversampling
    OVERSAMPLE = True
    
    # Logging
    LOG_LEVEL = "INFO"

# ============================================================================
# SETUP DE REPRODUTIBILIDADE
# ============================================================================
def setup_seed(seed=42):
    """Fixar todas as seeds para reprodutibilidade total"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Pode ser True para performance, mas mantemos False para reprodutibilidade
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# ============================================================================
# TRANSFORMAÇÕES 3D
# ============================================================================
class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class RandomFlip3D:
    def __init__(self, p=0.5, axis=(1, 2)):
        self.p = p
        self.axis = axis  # (height, width) para volumes 3D
    
    def __call__(self, volume):
        if random.random() < self.p:
            volume = torch.flip(volume, dims=[self.axis[0]])
        if random.random() < self.p:
            volume = torch.flip(volume, dims=[self.axis[1]])
        return volume

class RandomRotation3D:
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, volume):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            # Rotação apenas no plano axial (dimensões H e W)
            volume = F.interpolate(
                volume.unsqueeze(0),
                size=volume.shape[1:],
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
        return volume

class Normalize3D:
    """Normalização min-max por volume individual"""
    def __call__(self, volume):
        min_val = volume.min()
        max_val = volume.max()
        if (max_val - min_val) > 1e-8:
            volume = (volume - min_val) / (max_val - min_val)
        return volume

def get_transforms(augment=True):
    """Obter transformações para dados 3D"""
    transforms = [Normalize3D()]
    
    if augment:
        transforms.extend([
            RandomFlip3D(p=0.5, axis=(2, 3)),  # Flip nas dimensões H e W
        ])
    
    return Compose3D(transforms)

# ============================================================================
# DATASET 3D
# ============================================================================
class KneeMRIDataset3D(Dataset):
    def __init__(self, normal_path, abnormal_path, transform=None):
        # Carregar dados com mmap para economizar RAM
        self.normal_data = np.load(normal_path, mmap_mode='r')
        self.abnormal_data = np.load(abnormal_path, mmap_mode='r')
        
        self.total_normal = len(self.normal_data)
        self.total_abnormal = len(self.abnormal_data)
        
        # Labels: 0 = normal, 1 = abnormal
        self.labels = np.concatenate([
            np.zeros(self.total_normal, dtype=np.int32),
            np.ones(self.total_abnormal, dtype=np.int32)
        ])
        
        self.transform = transform
        
        print(f"Dataset carregado: {self.total_normal} normais, {self.total_abnormal} anormais")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if idx < self.total_normal:
            volume = self.normal_data[idx].astype(np.float32)
            label = 0
        else:
            volume = self.abnormal_data[idx - self.total_normal].astype(np.float32)
            label = 1
        
        # Converter para tensor (1, D, H, W)
        volume = torch.from_numpy(volume).unsqueeze(0).float()
        
        if self.transform:
            volume = self.transform(volume)
        
        return volume, torch.tensor(label, dtype=torch.float32)
    
    def get_class_weights(self):
        """Calcular peso para classe positiva automaticamente"""
        n_normal = self.total_normal
        n_abnormal = self.total_abnormal
        
        if n_abnormal == 0:
            return torch.tensor([1.0])
        
        # pos_weight = n_negative / n_positive
        pos_weight = torch.tensor([n_normal / n_abnormal])
        return pos_weight

# ============================================================================
# MODELO R3D-18
# ============================================================================
def create_model(pretrained=True, device='cuda'):
    """Criar modelo R3D-18 para classificação binária"""
    if pretrained:
        print("📥 Carregando R3D-18 pré-treinado (Kinetics-400)")
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
    else:
        print("🆕 Inicializando R3D-18 do zero")
        model = r3d_18(weights=None)
    
    # Substituir última camada para classificação binária
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    
    # Inicialização Xavier para a nova camada
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    
    model = model.to(device)
    
    # Log de parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Parâmetros: Total={total_params:,}, Treináveis={trainable_params:,}")
    
    return model

# ============================================================================
# FUNÇÕES DE TREINAMENTO COM AMP
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device, amp=True):
    """Treinar uma época"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Treino', leave=False)
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # R3D-18 espera 3 canais (repetir canal único)
        inputs = inputs.repeat(1, 3, 1, 1, 1)
        
        optimizer.zero_grad()
        
        # Forward com AMP
        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Estatísticas
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Atualizar barra
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total if total > 0 else 0
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
    
    return total_loss / len(loader), accuracy

def validate_epoch(model, loader, criterion, device):
    """Validar uma época"""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validação', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.repeat(1, 3, 1, 1, 1)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())
            
            avg_loss = total_loss / len(all_labels) if all_labels else 0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    all_probs = 1 / (1 + np.exp(-all_logits))  # Sigmoid
    
    return total_loss / len(loader), all_labels, all_probs

# ============================================================================
# CÁLCULO DE MÉTRICAS
# ============================================================================
def calculate_metrics(labels, probs, threshold=0.5):
    """Calcular todas as métricas científicas"""
    preds = (probs >= threshold).astype(int)
    
    # Métricas básicas
    accuracy = accuracy_score(labels, preds) * 100
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(labels, probs)
    except ValueError:
        auc_roc = 0.0
    
    # F1-Score
    f1 = f1_score(labels, preds, zero_division=0)
    
    # AUC-PR
    precision, recall, _ = precision_recall_curve(labels, probs)
    auc_pr = auc(recall, precision)
    
    # Matriz de confusão
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    # Métricas específicas
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'f1_score': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(ppv),
        'negative_predictive_value': float(npv),
        'n_samples': int(len(labels)),
        'n_positive': int(np.sum(labels)),
        'n_negative': int(np.sum(1 - labels)),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    }

def save_roc_data(labels, probs, save_path):
    """Salvar dados da curva ROC para consistência"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc': float(auc_score)
    }
    
    with open(save_path, 'w') as f:
        json.dump(roc_data, f, indent=4)

# ============================================================================
# TREINAMENTO POR FOLD
# ============================================================================
def train_fold(model, train_loader, val_loader, dataset, config, 
               fold_idx, fold_dir, device):
    """Treinar um fold completo"""
    # Calcular pos_weight automaticamente
    if config.POS_WEIGHT_AUTO:
        pos_weight = dataset.get_class_weights().to(device)
        print(f"   📊 pos_weight: {pos_weight.item():.2f}")
    else:
        pos_weight = torch.tensor([2.0]).to(device)
    
    # Loss function (BCEWithLogitsLoss padrão)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Otimizador
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # GradScaler para AMP
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    
    # Early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_epoch = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_auc': [], 'epochs': []
    }
    
    # Barra de progresso do fold
    fold_pbar = tqdm(range(config.NUM_EPOCHS), 
                     desc=f'Fold {fold_idx+1}', 
                     position=0, leave=True)
    
    for epoch in fold_pbar:
        # Treino
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            scaler, device, config.AMP
        )
        
        # Validação
        val_loss, val_labels, val_probs = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Calcular métricas
        val_metrics = calculate_metrics(val_labels, val_probs)
        val_acc = val_metrics['accuracy']
        val_auc = val_metrics['auc_roc']
        
        # Atualizar histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['epochs'].append(epoch)
        
        # Atualizar barra
        fold_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.2f}%',
            'val_auc': f'{val_auc:.4f}'
        })
        
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            best_metrics = val_metrics
            
            # Salvar melhor modelo
            torch.save(model.state_dict(), fold_dir / 'best_model.pth')
            
            # Salvar y_true e y_prob para consistência AUC
            np.save(fold_dir / 'y_true.npy', val_labels)
            np.save(fold_dir / 'y_prob.npy', val_probs)
            
            # Salvar dados da ROC
            save_roc_data(val_labels, val_probs, fold_dir / 'roc.json')
            
            print(f"      💾 Modelo salvo (epoch {epoch+1}, AUC={val_auc:.4f})")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= config.EARLY_STOP_PATIENCE:
            print(f"      ⏹️  Early stopping (paciência {config.EARLY_STOP_PATIENCE})")
            break
    
    fold_pbar.close()
    
    # Carregar melhor modelo
    model.load_state_dict(torch.load(fold_dir / 'best_model.pth'))
    
    # Salvar histórico
    with open(fold_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Salvar métricas
    with open(fold_dir / 'metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=4)
    
    print(f"      ✅ Fold {fold_idx+1} concluído (melhor epoch {best_epoch+1})")
    print(f"      📊 Acurácia: {best_metrics['accuracy']:.2f}%, "
          f"AUC: {best_metrics['auc_roc']:.4f}, "
          f"F1: {best_metrics['f1_score']:.4f}")
    
    return best_metrics

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================
def main():
    print("=" * 100)
    print("🎓 EXPERIMENTO 01: 5-fold CV com R3D-18 pré-treinado")
    print("   Thalles Cotta Fontainha - PPGIO CEFET/RJ DEZ/2025")
    print("=" * 100)
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Sistema: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print("=" * 100)
    
    # Setup inicial
    setup_seed(Config.SEED)
    
    # Criar diretório de saída
    exp_dir = Config.OUTPUT_DIR / Config.EXP_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar logging
    log_file = exp_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {vram:.2f} GB")
        logger.info(f"AMP: {Config.AMP} (batch_size={Config.BATCH_SIZE})")
        
        # Verificar se batch_size é seguro
        if Config.BATCH_SIZE > 4 and not Config.AMP:
            logger.warning("⚠️  Batch_size grande sem AMP pode estourar VRAM!")
    else:
        logger.warning("⚠️  Sem GPU - treinamento será lento")
        Config.BATCH_SIZE = 2
        Config.AMP = False
    
    # Salvar configuração (corrigido para serializar Path objects)
    config_dict = {}
    for k, v in Config.__dict__.items():
        if not k.startswith('_') and k != 'device':
            # Converter Path objects para string
            if isinstance(v, Path):
                config_dict[k] = str(v)
            else:
                config_dict[k] = v
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Carregar dataset
    normal_path = Config.DATA_DIR / Config.NORMAL_FILE
    abnormal_path = Config.DATA_DIR / Config.ABNORMAL_FILE
    
    logger.info(f"Carregando dados:")
    logger.info(f"  Normal: {normal_path}")
    logger.info(f"  Abnormal: {abnormal_path}")
    
    if not normal_path.exists() or not abnormal_path.exists():
        logger.error("❌ Arquivos de dados não encontrados!")
        return
    
    transforms = get_transforms(Config.AUGMENTATION)
    dataset = KneeMRIDataset3D(
        str(normal_path), 
        str(abnormal_path), 
        transform=transforms
    )
    
    logger.info(f"Dataset: {len(dataset)} amostras")
    logger.info(f"  Normais: {dataset.total_normal}")
    logger.info(f"  Anormais: {dataset.total_abnormal}")
    
    # Criar folds estratificados
    skf = StratifiedKFold(
        n_splits=Config.N_SPLITS, 
        shuffle=True, 
        random_state=Config.SEED
    )
    
    folds = list(skf.split(range(len(dataset)), dataset.labels))
    logger.info(f"\nCriando {Config.N_SPLITS}-fold Stratified CV")
    
    # Resultados
    fold_results = []
    fold_metrics = []
    
    print("\n" + "=" * 100)
    print("🚀 INICIANDO 5-FOLD CROSS-VALIDATION")
    print("=" * 100)
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        start_time = time.time()
        
        print(f"\n🔄 FOLD {fold_idx + 1}/{Config.N_SPLITS}")
        print(f"   Treino: {len(train_idx)} amostras")
        print(f"   Validação: {len(val_idx)} amostras")
        
        # Oversampling apenas no treino
        if Config.OVERSAMPLE:
            ros = RandomOverSampler(random_state=Config.SEED + fold_idx)
            train_idx_reshaped = np.array(train_idx).reshape(-1, 1)
            train_labels = dataset.labels[train_idx]
            
            resampled_idx, _ = ros.fit_resample(train_idx_reshaped, train_labels)
            train_idx = resampled_idx.flatten().tolist()
            
            logger.info(f"Fold {fold_idx+1}: Oversampling aplicado")
            print(f"   Após oversampling: {len(train_idx)} amostras")
        
        # Criar DataLoaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=True
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )
        
        # Criar diretório do fold
        fold_dir = exp_dir / f"fold_{fold_idx+1}"
        fold_dir.mkdir(exist_ok=True)
        
        # Criar modelo
        model = create_model(pretrained=Config.PRETRAINED, device=device)
        
        # Treinar fold
        metrics = train_fold(
            model, train_loader, val_loader, dataset,
            Config, fold_idx, fold_dir, device
        )
        
        # Salvar resultados do fold
        fold_duration = time.time() - start_time
        fold_result = {
            'fold': fold_idx + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'duration_minutes': fold_duration / 60,
            'metrics': metrics
        }
        
        with open(fold_dir / 'fold_result.json', 'w') as f:
            json.dump(fold_result, f, indent=4)
        
        fold_results.append(fold_result)
        fold_metrics.append(metrics)
        
        print(f"   ✅ Fold concluído em {fold_duration/60:.1f} minutos")
    
    # ========================================================================
    # RELATÓRIO FINAL
    # ========================================================================
    print("\n" + "=" * 100)
    print("📊 RELATÓRIO FINAL - EXPERIMENTO 01")
    print("=" * 100)
    
    if fold_metrics:
        # Calcular médias e desvios
        accuracies = [m['accuracy'] for m in fold_metrics]
        aucs = [m['auc_roc'] for m in fold_metrics]
        f1s = [m['f1_score'] for m in fold_metrics]
        sensitivities = [m['sensitivity'] for m in fold_metrics]
        specificities = [m['specificity'] for m in fold_metrics]
        
        print(f"\n📈 MÉTRICAS (Média ± Desvio Padrão):")
        print(f"   Acurácia:    {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}")
        print(f"   AUC-ROC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"   F1-Score:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"   Sensitividade: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
        print(f"   Especificidade: {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")
        
        # Resultados por fold
        print(f"\n📋 DETALHES POR FOLD:")
        for i, metrics in enumerate(fold_metrics):
            print(f"   Fold {i+1}: Acc={metrics['accuracy']:.2f}%, "
                  f"AUC={metrics['auc_roc']:.4f}, "
                  f"F1={metrics['f1_score']:.4f}")
        
        # Concatenar todas as predições para curva ROC global
        all_labels = []
        all_probs = []
        for i in range(Config.N_SPLITS):
            fold_dir = exp_dir / f"fold_{i+1}"
            y_true = np.load(fold_dir / 'y_true.npy')
            y_prob = np.load(fold_dir / 'y_prob.npy')
            all_labels.append(y_true)
            all_probs.append(y_prob)
        
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        np.save(exp_dir / 'all_labels.npy', all_labels)
        np.save(exp_dir / 'all_probs.npy', all_probs)
        
        # Calcular métricas globais
        global_metrics = calculate_metrics(all_labels, all_probs)
        save_roc_data(all_labels, all_probs, exp_dir / 'global_roc.json')
        
        print(f"\n🌍 MÉTRICAS GLOBAIS (todos os folds):")
        print(f"   Acurácia:    {global_metrics['accuracy']:.2f}%")
        print(f"   AUC-ROC:     {global_metrics['auc_roc']:.4f}")
        print(f"   F1-Score:    {global_metrics['f1_score']:.4f}")
        print(f"   Amostras:    {global_metrics['n_samples']}")
        
        # Salvar sumário final
        summary = {
            'experiment': Config.EXP_NAME,
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': platform.python_version(),
                'pytorch_version': torch.__version__,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'seed': Config.SEED
            },
            'config': config_dict,
            'fold_results': fold_results,
            'average_metrics': {
                'accuracy': {'mean': float(np.mean(accuracies)), 'std': float(np.std(accuracies))},
                'auc_roc': {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))},
                'f1_score': {'mean': float(np.mean(f1s)), 'std': float(np.std(f1s))},
                'sensitivity': {'mean': float(np.mean(sensitivities)), 'std': float(np.std(sensitivities))},
                'specificity': {'mean': float(np.mean(specificities)), 'std': float(np.std(specificities))}
            },
            'global_metrics': global_metrics,
            'dataset_info': {
                'total_samples': len(dataset),
                'normal_samples': dataset.total_normal,
                'abnormal_samples': dataset.total_abnormal,
                'class_ratio': dataset.total_abnormal / len(dataset)
            }
        }
        
        with open(exp_dir / 'experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n💾 Resultados salvos em: {exp_dir}")
        
        # Gerar relatório em formato de tabela
        print(f"\n📋 RESUMO PARA ARTIGO (latex-ready):")
        print("\\begin{table}[ht]")
        print("\\centering")
        print("\\caption{Resultados da validação cruzada 5-fold com R3D-18 pré-treinado}")
        print("\\label{tab:cv_results_pretrained}")
        print("\\begin{tabular}{lccccc}")
        print("\\hline")
        print("Fold & Acurácia (\\%) & AUC-ROC & F1-Score & Sensitividade & Especificidade \\\\")
        print("\\hline")
        for i, m in enumerate(fold_metrics):
            print(f"{i+1} & {m['accuracy']:.2f} & {m['auc_roc']:.4f} & {m['f1_score']:.4f} & {m['sensitivity']:.4f} & {m['specificity']:.4f} \\\\")
        print("\\hline")
        print(f"Média & {np.mean(accuracies):.2f} & {np.mean(aucs):.4f} & {np.mean(f1s):.4f} & {np.mean(sensitivities):.4f} & {np.mean(specificities):.4f} \\\\")
        print(f"Desvio & {np.std(accuracies):.2f} & {np.std(aucs):.4f} & {np.std(f1s):.4f} & {np.std(sensitivities):.4f} & {np.std(specificities):.4f} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
    
    print("\n" + "=" * 100)
    print("🎉 EXPERIMENTO 01 CONCLUÍDO COM SUCESSO!")
    print("=" * 100)

if __name__ == '__main__':
    main()