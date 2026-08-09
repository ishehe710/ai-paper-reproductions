baseline:
    Test Results
	Discriminator:  1.8490784399091353e-06
	Generator:      -2.3841860752327193e-07

    Epoch 22/22:
	Discriminator Train   Loss: 0.0000
	Generator Train       Loss: -0.0000
	Discriminator Val     Loss: 0.0000
	Generator Val         Loss: -0.0000
	D Learning Rate:   0.09999120
	G Learning Rate:   0.09999120
	D Momentum:        0.5169
	G Momentum:        0.5169
    time: 24m 5s

ablation 1: saturated loss
    Test Results
	Discriminator:  5.566468963115767e-06
	Generator:      -1.4901171425663051e-06

    Discriminator Train   Loss: 0.0000
	Generator Train       Loss: 13.3585
	Discriminator Val     Loss: 0.0000
	Generator Val         Loss: 13.4332
	D Learning Rate:   0.09999120
	G Learning Rate:   0.09999120
	D Momentum:        0.5169
	G Momentum:        0.5169
    time: 107m 6s

ablation 2: saturated loss with 240->120
    Test Results
	Discriminator:  5.794414085903554e-06
	Generator:      -1.3709077393286861e-06

    Epoch 22/22:
	Discriminator Train   Loss: 0.0000
	Generator Train       Loss: 13.6391
	Discriminator Val     Loss: 0.0000
	Generator Val         Loss: 13.5184
	D Learning Rate:   0.09999120
	G Learning Rate:   0.09999120
	D Momentum:        0.5169
	G Momentum:        0.5169
    time: 37m 2s

ablation 3: saturated loss with Adam optimizer

    Test Results
	Discriminator:  1.5497212473292164e-10
	Generator:      0.0

    Epoch 22/22:
	Discriminator Train   Loss: 0.0000
	Generator Train       Loss: 26.3774
	Discriminator Val     Loss: 0.0000
	Generator Val         Loss: 26.5853

    time: 157m 2s

| Experiment     | Modification               | D Test Loss | G Test Loss | Final G Val Loss | Time |
| -------------- | -------------------------- | ----------: | ----------: | ---------------: | ---: |
| **Baseline**   | Paper setup                |     1.85e-6 |          ~0 |               ~0 |  24m |
| **Ablation 1** | Non-saturating G loss      |     5.57e-6 |          ~0 |        **13.43** | 107m |
| **Ablation 2** | Non-saturating + D 240→120 |     5.79e-6 |          ~0 |        **13.52** |  37m |
| **Ablation 3** | Non-saturating + Adam      |    1.55e-10 |           0 |        **26.59** | 157m |
