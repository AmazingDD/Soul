

# test
model = model_map[args.model_type](num_classes=num_classes, T=args.T, input_shape=input_shape)
model.load_state_dict(
    torch.load(f'{args.dataset}_{args.model}_T{args.T}_thr{args.flat_width}_seed{args.seed}_ckpt_best.pth', map_location='cpu'))

# latency


# SOPs theoretical energy cost



# real energy cost 