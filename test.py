
import torch
import torch.nn as nn
import onnx

# Simple model with a loop
class SimpleLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        for i in range(num_steps):
            x = self.linear(x) + 1.0
        return x

# Regular model with loop (non-scripted)
class RegularLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x, num_steps):
        for i in range(num_steps):
            x = self.linear(x) + 1.0
        return x

# Test both versions
def test_exports():
    # Create models
    regular_model = SimpleLoop()
    scripted_model = torch.jit.script(regular_model)  # Script the instance
    
    # Sample input
    x = torch.randn(1, 10)
    num_steps = 5
    
    print("Testing TorchScript version:")
    try:
        torch.onnx.export(
            scripted_model,
            (x, num_steps),
            "scripted_loop.onnx",
            input_names=['x', 'num_steps'],
            output_names=['output'],
            dynamic_axes={'x': {0: 'batch_size'}},
            opset_version=11
        )
        print("✅ Scripted model exported successfully")
    except Exception as e:
        print(f"❌ Scripted model failed: {e}")
    
    print("\nTesting regular model:")
    try:
        torch.onnx.export(
            regular_model,
            (x, num_steps),
            "regular_loop.onnx", 
            input_names=['x', 'num_steps'],
            output_names=['output'],
            dynamic_axes={'x': {0: 'batch_size'}},
            opset_version=11
        )
        print("✅ Regular model exported successfully")
    except Exception as e:
        print(f"❌ Regular model failed: {e}")
    
    # Analyze the ONNX graphs
    print("\n" + "="*50)
    print("ONNX GRAPH ANALYSIS")
    print("="*50)
    
    try:
        # Load and analyze scripted version
        scripted_onnx = onnx.load("scripted_loop.onnx")
        print(f"\nScripted model nodes: {len(scripted_onnx.graph.node)}")
        print("Node types:")
        for i, node in enumerate(scripted_onnx.graph.node):
            print(f"  {i}: {node.op_type}")
    except:
        print("Could not analyze scripted model")
    
    try:
        # Load and analyze regular version  
        regular_onnx = onnx.load("regular_loop.onnx")
        print(f"\nRegular model nodes: {len(regular_onnx.graph.node)}")
        print("Node types:")
        for i, node in enumerate(regular_onnx.graph.node):
            print(f"  {i}: {node.op_type}")
    except:
        print("Could not analyze regular model")

if __name__ == "__main__":
    test_exports()