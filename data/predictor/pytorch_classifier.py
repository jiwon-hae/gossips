#!/usr/bin/env python3
"""
PyTorch-based classifier for celebrity news events.
Implements neural network models with embeddings for text classification.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# PyTorch libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam, SGD
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import pickle
except ImportError:
    print("Warning: PyTorch not installed. Install with: pip install torch")

from data.collection.models import NewsArticle, TrainingDataSample, ClassificationResult
from data.collection.config import EventCategory
from data.predictor.ml_classifier import ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class PyTorchTrainingConfig:
    """Configuration for PyTorch training."""
    model_type: str = "mlp"  # mlp, lstm, transformer
    embedding_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    weight_decay: float = 1e-4
    
    # Data parameters
    max_vocab_size: int = 10000
    max_sequence_length: int = 100
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Training settings
    early_stopping_patience: int = 10
    lr_scheduler: str = "plateau"  # step, plateau, none
    
    # Device and save settings
    device: str = "auto"  # auto, cpu, cuda
    save_model: bool = True
    model_save_path: str = f"{project_root}/models/celebrity_classifier_pytorch.pth"
    random_state: int = 42


class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], 
                 max_length: int = 100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = self._tokenize(text)
        indices = self._tokens_to_indices(tokens)
        
        # Pad or truncate to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([0] * (self.max_length - len(indices)))  # 0 is padding token
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """Convert tokens to vocabulary indices."""
        return [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_classes: int, dropout: float = 0.3):
        super(MLPClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # MLP layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # Average pooling over sequence dimension
        pooled = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)
        
        # MLP layers
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Classification layer
        x = self.dropout(last_hidden)
        x = self.fc(x)
        
        return x


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_classes: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.3):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        seq_len = x.size(1)
        
        # Embedding + positional encoding
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        embedded += self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Create padding mask
        padding_mask = (x == 0)  # True for padding tokens
        
        # Transformer forward pass
        transformer_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Global average pooling (excluding padding tokens)
        mask = (~padding_mask).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        pooled = (transformer_out * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Classification layer
        x = self.dropout(pooled)
        x = self.fc(x)
        
        return x


class EventPyTorchClassifier:
    """PyTorch-based classifier for celebrity news events."""
    
    def __init__(self, config: PyTorchTrainingConfig = None):
        """Initialize the PyTorch classifier."""
        self.config = config or PyTorchTrainingConfig()
        self.model: Optional[nn.Module] = None
        self.vocab: Optional[Dict[str, int]] = None
        self.label_encoder = LabelEncoder()
        self.metrics: Optional[ModelMetrics] = None
        self.is_trained = False
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        
        # Create models directory if it doesn't exist
        model_dir = os.path.dirname(self.config.model_save_path)
        if model_dir:  # Only create if there's a directory path
            os.makedirs(model_dir, exist_ok=True)
    
    def _build_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from training texts."""
        word_counts = Counter()
        
        for text in texts:
            tokens = text.lower().split()
            word_counts.update(tokens)
        
        # Create vocabulary with special tokens
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # Add most frequent words up to max_vocab_size
        most_common = word_counts.most_common(self.config.max_vocab_size - 2)
        for word, _ in most_common:
            vocab[word] = len(vocab)
        
        logger.info(f"Built vocabulary with {len(vocab)} words")
        return vocab
    
    def _create_model(self, num_classes: int) -> nn.Module:
        """Create the specified model architecture."""
        vocab_size = len(self.vocab)
        
        if self.config.model_type == "mlp":
            model = MLPClassifier(
                vocab_size=vocab_size,
                embedding_dim=self.config.embedding_dim,
                hidden_dim=self.config.hidden_dim,
                num_classes=num_classes,
                dropout=self.config.dropout
            )
        elif self.config.model_type == "lstm":
            model = LSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=self.config.embedding_dim,
                hidden_dim=self.config.hidden_dim,
                num_classes=num_classes,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        elif self.config.model_type == "transformer":
            model = TransformerClassifier(
                vocab_size=vocab_size,
                embedding_dim=self.config.embedding_dim,
                hidden_dim=self.config.hidden_dim,
                num_classes=num_classes,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        return model.to(self.device)
    
    def _prepare_training_data(self, training_samples: List[TrainingDataSample]) -> Tuple[List[str], List[str]]:
        """Prepare training data from samples."""
        texts = []
        labels = []
        
        for sample in training_samples:
            texts.append(sample.text)
            labels.append(sample.label.value)
        
        return texts, labels
    
    def train(self, training_samples: List[TrainingDataSample]) -> ModelMetrics:
        """Train the PyTorch classifier."""
        logger.info(f"Training {self.config.model_type} PyTorch classifier on {len(training_samples)} samples")
        
        # Prepare data
        texts, labels = self._prepare_training_data(training_samples)
        
        # Build vocabulary
        self.vocab = self._build_vocabulary(texts)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        # Check class distribution
        class_counts = Counter(labels)
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        # Check if stratification is possible
        min_class_count = min(class_counts.values())
        use_stratify = min_class_count >= 2
        
        if not use_stratify:
            logger.warning(f"Some classes have only {min_class_count} sample(s). Disabling stratification.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_encoded,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_encoded if use_stratify else None
        )
        
        # Further split training data for validation
        # Check if stratification is still possible after first split
        train_class_counts = Counter([self.label_encoder.inverse_transform([y])[0] for y in y_train])
        min_train_class_count = min(train_class_counts.values())
        use_val_stratify = min_train_class_count >= 2
        
        if not use_val_stratify:
            logger.warning(f"Training set has classes with only {min_train_class_count} sample(s). Disabling validation stratification.")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_size,
            random_state=self.config.random_state,
            stratify=y_train if use_val_stratify else None
        )
        
        # Create datasets and data loaders
        train_dataset = TextDataset(X_train, y_train, self.vocab, self.config.max_sequence_length)
        val_dataset = TextDataset(X_val, y_val, self.vocab, self.config.max_sequence_length)
        test_dataset = TextDataset(X_test, y_test, self.vocab, self.config.max_sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Create model
        self.model = self._create_model(num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate, 
                        weight_decay=self.config.weight_decay)
        
        # Learning rate scheduler
        if self.config.lr_scheduler == "step":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.config.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        else:
            scheduler = None
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_acc = self._evaluate(val_loader)
            val_accuracies.append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            if scheduler:
                if self.config.lr_scheduler == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                if self.config.save_model:
                    self._save_checkpoint()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Ensure we save the final model if no checkpoint was saved yet
        if self.config.save_model and not os.path.exists(self.config.model_save_path):
            self._save_checkpoint()
        
        # Load best model if checkpoint was saved
        if self.config.save_model and os.path.exists(self.config.model_save_path):
            self._load_checkpoint()
        
        # Final evaluation
        test_acc = self._evaluate(test_loader)
        
        # Generate detailed metrics
        self.metrics = self._generate_metrics(test_loader, len(X_train), len(X_test))
        
        self.is_trained = True
        
        logger.info(f"Training completed. Test Accuracy: {test_acc:.4f}")
        
        return self.metrics
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on given data loader."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total
    
    def _generate_metrics(self, test_loader: DataLoader, train_size: int, test_size: int) -> ModelMetrics:
        """Generate comprehensive metrics."""
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # Convert back to original labels
        y_true_labels = self.label_encoder.inverse_transform(y_true)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
        report_str = classification_report(y_true_labels, y_pred_labels)
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract per-class metrics
        precision = {k: v['precision'] for k, v in report.items() if isinstance(v, dict)}
        recall = {k: v['recall'] for k, v in report.items() if isinstance(v, dict)}
        f1 = {k: v['f1-score'] for k, v in report.items() if isinstance(v, dict)}
        support = {k: v['support'] for k, v in report.items() if isinstance(v, dict)}
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            support=support,
            confusion_matrix=cm,
            classification_report=report_str,
            cross_val_scores=[accuracy],  # Single validation score for now
            training_samples=train_size,
            test_samples=test_size
        )
    
    def predict(self, texts: List[str]) -> List[ClassificationResult]:
        """Predict labels for new texts."""
        if not self.is_trained or not self.model:
            raise ValueError("Model must be trained before making predictions")
        
        # Create dataset for prediction
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        dataset = TextDataset(texts, dummy_labels, self.vocab, self.config.max_sequence_length)
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                
                all_predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        results = []
        for i, text in enumerate(texts):
            predicted_class_encoded = all_predictions[i]
            predicted_class = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
            confidence = all_probabilities[i].max()
            
            # Get all class probabilities
            all_probs = dict(zip(
                self.label_encoder.inverse_transform(range(len(all_probabilities[i]))),
                all_probabilities[i]
            ))
            
            # Convert to EventCategory
            try:
                category = EventCategory(predicted_class)
            except ValueError:
                category = EventCategory.OTHER
            
            result = ClassificationResult(
                category=category,
                confidence=float(confidence),
                method="pytorch_classifier",
                raw_scores={EventCategory(k): float(v) for k, v in all_probs.items() if k in [e.value for e in EventCategory]},
                explanation=f"PyTorch {self.config.model_type} prediction"
            )
            
            results.append(result)
        
        return results
    
    def predict_article(self, article: NewsArticle) -> ClassificationResult:
        """Predict category for a single article."""
        text = article.title
        if article.summary:
            text += " " + article.summary
        
        results = self.predict([text])
        return results[0]
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'vocab': self.vocab,
            'label_encoder': self.label_encoder,
            'trained_at': datetime.now()
        }
        torch.save(checkpoint, self.config.model_save_path)
        logger.info(f"Model checkpoint saved to {self.config.model_save_path}")
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        if not os.path.exists(self.config.model_save_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_save_path}")
        
        try:
            # Load with weights_only=False for backward compatibility
            checkpoint = torch.load(self.config.model_save_path, map_location=self.device, weights_only=False)
            
            # Check if the model architecture is compatible
            current_vocab_size = len(self.vocab) if self.vocab else 0
            current_num_classes = len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 0
            
            # Try to get saved vocab and classes info
            saved_vocab_size = len(checkpoint.get('vocab', {}))
            saved_label_encoder = checkpoint.get('label_encoder')
            saved_num_classes = len(saved_label_encoder.classes_) if saved_label_encoder and hasattr(saved_label_encoder, 'classes_') else 0
            
            if current_vocab_size > 0 and saved_vocab_size > 0 and current_vocab_size != saved_vocab_size:
                logger.warning(f"Vocabulary size mismatch: current={current_vocab_size}, saved={saved_vocab_size}. Skipping checkpoint load.")
                return
            
            if current_num_classes > 0 and saved_num_classes > 0 and current_num_classes != saved_num_classes:
                logger.warning(f"Number of classes mismatch: current={current_num_classes}, saved={saved_num_classes}. Skipping checkpoint load.")
                return
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.vocab = checkpoint['vocab']
            self.label_encoder = checkpoint['label_encoder']
            
            logger.info(f"Model checkpoint loaded from {self.config.model_save_path}")
            
        except (RuntimeError, KeyError) as e:
            logger.warning(f"Failed to load checkpoint due to architecture mismatch: {e}")
            logger.info("Continuing with newly initialized model")
    
    def save_model(self, path: str = None):
        """Save the complete trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = path or self.config.model_save_path
        self._save_checkpoint()
    
    def load_model(self, path: str = None):
        """Load a previously trained model."""
        load_path = path or self.config.model_save_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        try:
            # Load with weights_only=False for backward compatibility
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
            
            # Restore configuration and vocabulary
            self.vocab = checkpoint['vocab']
            self.label_encoder = checkpoint['label_encoder']
            
            # Create model with correct architecture
            num_classes = len(self.label_encoder.classes_)
            self.model = self._create_model(num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.is_trained = True
            
            logger.info(f"Model loaded from {load_path}")
            logger.info(f"Model trained at: {checkpoint.get('trained_at', 'Unknown')}")
            
        except (RuntimeError, KeyError) as e:
            logger.error(f"Failed to load model from {load_path}: {e}")
            raise
    
    def print_evaluation_report(self):
        """Print detailed evaluation report."""
        if not self.metrics:
            print("No evaluation metrics available. Train the model first.")
            return
        
        print("=" * 60)
        print("CELEBRITY NEWS PYTORCH CLASSIFIER - EVALUATION REPORT")
        print("=" * 60)
        
        print(f"Model Type: {self.config.model_type}")
        print(f"Device: {self.device}")
        print(f"Training Samples: {self.metrics.training_samples}")
        print(f"Test Samples: {self.metrics.test_samples}")
        print(f"Overall Accuracy: {self.metrics.accuracy:.3f}")
        
        print("\nPer-Class Metrics:")
        print("-" * 40)
        for class_name in self.metrics.precision.keys():
            if class_name in ['macro avg', 'weighted avg']:
                continue
            precision = self.metrics.precision.get(class_name, 0)
            recall = self.metrics.recall.get(class_name, 0)
            f1 = self.metrics.f1_score.get(class_name, 0)
            support = self.metrics.support.get(class_name, 0)
            print(f"{class_name:15}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} Support={support}")
        
        print("\nModel Configuration:")
        print("-" * 40)
        print(f"Embedding Dim: {self.config.embedding_dim}")
        print(f"Hidden Dim: {self.config.hidden_dim}")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Learning Rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_epochs}")
        
        print("\nDetailed Classification Report:")
        print("-" * 40)
        print(self.metrics.classification_report)