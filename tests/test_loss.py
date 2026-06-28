import unittest
import torch
from vnl_loss import VirtualNormalLoss

class TestVirtualNormalLoss(unittest.TestCase):
    
    def test_vnl_forward_and_backward(self):
        # 1. Mock camera intrinsics
        vnl = VirtualNormalLoss(focal_x=700.0, focal_y=700.0, center_x=320.0, center_y=240.0, num_samples=100)
        
        # 2. Create fake depth maps (Batch=2, Channels=1, H=480, W=640)
        # We set requires_grad=True to test backpropagation
        pred = torch.rand((2, 1, 480, 640), requires_grad=True)
        gt = torch.rand((2, 1, 480, 640))
        
        # 3. Calculate Loss
        loss = vnl(pred, gt)
        
        # 4. Assert loss is a valid scalar
        self.assertFalse(torch.isnan(loss).item(), "Loss resulted in NaN!")
        self.assertGreaterEqual(loss.item(), 0.0, "Loss cannot be negative.")
        
        # 5. Assert backpropagation works (crucial for training)
        loss.backward()
        self.assertIsNotNone(pred.grad, "Gradients were not calculated!")

if __name__ == "__main__":
    unittest.main()