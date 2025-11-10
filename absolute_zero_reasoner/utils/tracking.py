from typing import Union, List

from verl.utils.tracking import Tracking


class ReasonRLTracking(Tracking):
    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None, resume='never', run_id=None, tags: List[str] = None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            wandb_kwargs = {}
            if resume == 'must':
                wandb_kwargs = {'resume': 'must', 'id': run_id}
            elif resume == 'allow':
                wandb_kwargs = {'resume': 'allow', 'id': run_id}
            if tags is not None:
                wandb_kwargs['tags'] = tags
            run = wandb.init(project=project_name, settings=wandb.Settings(start_method="thread"), name=experiment_name, config=config, **wandb_kwargs)
            self.run_id = run.id
            self.logger['wandb'] = wandb

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger
