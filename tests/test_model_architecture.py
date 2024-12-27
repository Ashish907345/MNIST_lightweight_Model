import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import Net
import pytest

class TestModelArchitecture:
    @pytest.fixture(scope="class")
    def model(self):
        return Net()

    def test_parameter_count(self, model):
        """Test if model has less than 20K parameters"""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params}")
        assert total_params < 20000, f"Model has too many parameters: {total_params}"

    def test_batch_normalization(self, model):
        """Test if model uses batch normalization"""
        has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
        assert has_bn, "Model should use BatchNormalization"

    def test_dropout(self, model):
        """Test if model uses dropout"""
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
        assert has_dropout, "Model should use Dropout"

    def test_architecture_type(self, model):
        """Test if model uses GAP instead of FC layers"""
        has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
        has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
        
        if has_gap:
            print("\nModel uses Global Average Pooling")
            assert not has_fc, "Model should not use both GAP and FC layers"
        else:
            print("\nModel uses Fully Connected layers")
            assert has_fc, "Model should use either GAP or FC layers" 