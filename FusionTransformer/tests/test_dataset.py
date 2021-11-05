import torch
#https://krokotsch.eu/cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html
def test_batch_independence(self):
    inputs = torch.randn(4, 1, 32, 32)
    inputs.requires_grad = True
    net = model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)

    # Compute forward pass in eval mode to deactivate batch norm
    net.eval()
    outputs = net(inputs)
    net.train()

    # Mask loss for certain samples in batch
    batch_size = inputs[0].shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(outputs)
    mask[mask_idx] = 0
    outputs = outputs * mask

    # Compute backward pass
    loss = outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(inputs.grad):
        if i == mask_idx:
            self.assertTrue(torch.all(grad == 0).item())
        else:
            self.assertTrue(not torch.all(grad == 0))



def test_all_parameters_updated(self):
    net = model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)
    optim = torch.optim.SGD(net.parameters(), lr=0.1)

    outputs = net(torch.randn(4, 1, 32, 32))
    loss = outputs.mean()
    loss.backward()
    optim.step()

    for param_name, param in self.net.named_parameters():
        if param.requires_grad:
            with self.subTest(name=param_name):
                self.assertIsNotNone(param.grad)
                self.assertNotEqual(0., torch.sum(param.grad ** 2))



def test_logging(self):
    with mock.patch.object(self.vae_trainer.summary, 'add_scalar') as add_scalar_mock:
        self.vae_trainer.train(1)

    expected_calls = [mock.call('train/recon_loss', mock.ANY, 0),
                      mock.call('train/kl_div_loss', mock.ANY, 0),
                      mock.call('train/loss', mock.ANY, 0),
                      mock.call('test/loss', mock.ANY, 0)]
    add_scalar_mock.assert_has_calls(expected_calls)

