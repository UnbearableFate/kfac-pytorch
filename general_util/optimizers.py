from typing import Callable, List, Optional, Tuple
import kfac
from torch.optim import AdamW
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR



lr = 0.001
weight_decay = 0.05
norm_weight_decay = 0.0
bias_weight_decay = 0.0
transformer_embedding_decay = 0.0
label_smoothing = 0.1
clip_grad_norm = 5.0

def get_swin_optimizer(model, args):
    parameters = set_weight_decay(
        model,
        weight_decay,
        norm_weight_decay=norm_weight_decay,
        custom_keys_weight_decay=[
            ('bias', bias_weight_decay),
            ('class_token', transformer_embedding_decay),
            ('position_embedding', transformer_embedding_decay),
            ('relative_position_bias_table', transformer_embedding_decay)
        ],
    )
    optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=1e-5)
    warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=args.lr_warmup_epochs)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs])
    return optimizer, lr_scheduler

def get_kfac_preconditioner(model, args,optimizer):
    preconditioner = kfac.preconditioner.KFACPreconditioner(
            model,
            factor_update_steps=args.kfac_factor_update_steps,
            inv_update_steps=args.kfac_inv_update_steps,
            damping=args.kfac_damping,
            factor_decay=args.kfac_factor_decay,
            kl_clip=args.kfac_kl_clip,
            lr=lambda x: optimizer.param_groups[0]['lr'],
            accumulation_steps=1,
            allreduce_bucket_cap_mb=25,
            colocate_factors=args.kfac_colocate_factors,
            compute_method= kfac.enums.ComputeMethod.EIGEN,
            grad_worker_fraction=kfac.enums.DistributedStrategy.COMM_OPT,
            grad_scaler=args.grad_scaler if 'grad_scaler' in args else None,
            skip_layers=args.kfac_skip_layers,
            train_method=args.train_com_method,
            is_packaged_send=True
        )

    def get_lambda(
        alpha: int,
        epochs: list[int] | None,
    ) -> Callable[[int], float]:
        """Create lambda function for param scheduler."""
        if epochs is None:
            _epochs = []
        else:
            _epochs = epochs

        def scale(epoch: int) -> float:
            """Compute current scale factor using epoch."""
            factor = 1.0
            for e in _epochs:
                if epoch >= e:
                    factor *= alpha
            return factor

        return scale

    kfac_param_scheduler = kfac.scheduler.LambdaParamScheduler(
        preconditioner,
        damping_lambda=get_lambda(
            args.kfac_damping_alpha,
            args.kfac_damping_decay,
        ),
        factor_update_steps_lambda=get_lambda(
            args.kfac_update_steps_alpha,
            args.kfac_update_steps_decay,
        ),
        inv_update_steps_lambda=get_lambda(
            args.kfac_update_steps_alpha,
            args.kfac_update_steps_decay,
        ),
    )

    return preconditioner, kfac_param_scheduler

def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups