init_wandb:
	uv run wandb login

download_models:
	modelscope download --model Qwen/Qwen2.5-Math-1.5B --local_dir models/qwen2.5-math-1.5b

upload_models:
	export WANDB_PROJECT=cs336-ass5
	uv run wandb artifact put artifacts/checkpoints/**/*.pt --type model --name checkpoints

upload_results:
	export WANDB_PROJECT=cs336-ass5
	uv run wandb artifact put artifacts/results/**/*.json --type results --name results

download_artifacts:
	uv run wandb artifact get checkpoints:latest
	uv run wandb artifact get results:latest