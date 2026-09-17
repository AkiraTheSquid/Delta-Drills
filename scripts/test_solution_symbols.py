"""Regression tests for authored module state versus required library syntax."""
import unittest
from solution_symbols import collect

class ModuleSymbols(unittest.TestCase):
    def test_authored_state_keeps_api_calls(self):
        symbols = collect('''import torch as t
class Scale(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = t.nn.Parameter(t.ones(2))
        self.register_buffer("offset", t.zeros(2))
    def forward(self, x):
        return (x-self.offset)*self.weight
''')
        self.assertTrue({'syntax.class', 'torch.nn.Module', 'torch.nn.Parameter',
                         'Tensor.register_buffer', 'builtin.super'} <= symbols)
        self.assertFalse({'Tensor.weight', 'Tensor.offset', 'torch.nn'} & symbols)

    def test_unassigned_framework_fields_remain(self):
        symbols = collect('''class Model(Base):
    def forward(self, x):
        return x.device, self.training, self.missing
''')
        self.assertTrue({'Tensor.device', 'Tensor.training', 'Tensor.missing'} <= symbols)

    def test_members_do_not_leak_between_classes(self):
        symbols = collect('''class A:
    def __init__(self):
        self.special = 1
class B:
    def forward(self, x):
        return self.special + x
''')
        self.assertIn('Tensor.special', symbols)

    def test_tensor_api_on_authored_field_remains(self):
        symbols = collect('''class A:
    def __init__(self, x):
        self.weight = x
    def forward(self, x):
        return self.weight.reshape(2, 3) + self.weight.grad
''')
        self.assertTrue({'Tensor.reshape', 'Tensor.grad'} <= symbols)
        self.assertNotIn('Tensor.weight', symbols)

    def test_namespace_call_is_qualified(self):
        symbols = collect('import torch as t\ny=t.nn.functional.conv2d(x,w)')
        self.assertIn('torch.nn.functional.conv2d', symbols)
        self.assertNotIn('torch.nn.functional', symbols)
        self.assertNotIn('torch.nn', symbols)

if __name__ == '__main__':
    unittest.main()
