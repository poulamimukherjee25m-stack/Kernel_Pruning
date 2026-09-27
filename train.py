"""
End-to-end training under the paper's unified rate-distortion objective
(Eq. 21):

    L = L_task(W_hat) + lambda_1 * L_dist + lambda_2 * R(K_eff)

There is no separate "train a baseline, then prune, then fine-tune" stage:
every convolution in the network is a PrototypeConv2d from the first
gradient step, and the compact dictionary structure is discovered *during*
this single training run, consistent with Section 0.1's reformulation of
compression as a representation-learning objective rather than a post-hoc
pruning procedure. After training, `compile_model` (Section 0.4) is applied
once, deterministically, to realize the learned structure as an actual
reduced-cost network -- that step performs no further optimization.

Supports three (dataset, architecture) pairs via --dataset/--arch:
  * cifar10       + resnet56  (default)
  * cifar100      + resnet56
  * tiny-imagenet + resnet50

Usage:
    python train.py --dataset cifar10 --arch resnet56 --epochs 200 \
        --batch-size 128 --lr 0.1 --lambda1 1.0 --lambda2 2e-4

    python train.py --dataset cifar100 --arch resnet56 --epochs 200 \
        --batch-size 128 --lr 0.1 --lambda1 1.0 --lambda2 2e-4

    python train.py --dataset tiny-imagenet --arch resnet50 \
        --data-root /path/to/tiny-imagenet-200 --epochs 100 \
        --batch-size 256 --lr 0.1 --lambda1 1.0 --lambda2 2e-4
"""
import argparse
import copy
import os
import time

import torch
import torch.nn as nn

from pruning import (
    cardinality_reg,
    compile_model,
    distortion_loss,
    effective_cardinality_report,
    profile_model,
    set_model_tau,
    anneal_tau,
    linear_warmup,
)
from pruning.proto_conv2d import PrototypeConv2d
from utils import AverageMeter, accuracy, set_seed

DATASET_DEFAULTS = {
    # (num_classes, default data-root, default input size, default batch size)
    "cifar10": dict(num_classes=10, data_root="./data/cifar10", input_size=32, batch_size=128),
    "cifar100": dict(num_classes=100, data_root="./data/cifar100", input_size=32, batch_size=128),
    "tiny-imagenet": dict(num_classes=200, data_root="./data/tiny-imagenet-200", input_size=64, batch_size=256),
}


def build_dataset(name, data_root, batch_size, workers):
    if name == "cifar10":
        from data.cifar import get_cifar10_loaders
        return get_cifar10_loaders(root=data_root, batch_size=batch_size, num_workers=workers)
    if name == "cifar100":
        from data.cifar100 import get_cifar100_loaders
        return get_cifar100_loaders(root=data_root, batch_size=batch_size, num_workers=workers)
    if name == "tiny-imagenet":
        from data.tiny_imagenet import get_tiny_imagenet_loaders
        return get_tiny_imagenet_loaders(root=data_root, batch_size=batch_size, num_workers=workers)
    raise ValueError(f"Unknown dataset: {name}")


def build_model(arch, num_classes, proto_kwargs):
    if arch == "resnet56":
        from models.resnet_cifar import resnet56_cifar
        return resnet56_cifar(num_classes=num_classes, proto_kwargs=proto_kwargs)
    if arch == "resnet50":
        from models.resnet_imagenet import resnet50_tiny_imagenet
        return resnet50_tiny_imagenet(num_classes=num_classes, proto_kwargs=proto_kwargs)
    raise ValueError(f"Unknown arch: {arch}")


def parse_args():
    p = argparse.ArgumentParser(description="Train CIFAR/Tiny-ImageNet ResNets with geometric-prototype pruning")
    # Dataset / architecture
    p.add_argument("--dataset", type=str, default="cifar10", choices=list(DATASET_DEFAULTS.keys()))
    p.add_argument("--arch", type=str, default="resnet56", choices=["resnet56", "resnet50"],
                    help="resnet56 (BasicBlock, for cifar10/cifar100) or resnet50 (Bottleneck, for tiny-imagenet)")
    # Data / optimization
    p.add_argument("--data-root", type=str, default=None,
                    help="defaults to a dataset-specific path (see DATASET_DEFAULTS) if omitted; "
                         "tiny-imagenet has no auto-download and MUST be pointed at an already-"
                         "extracted tiny-imagenet-200/ directory (see data/tiny_imagenet.py docstring)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=None, help="defaults to a dataset-specific value if omitted")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Rate-distortion objective (Eq. 21)
    p.add_argument("--lambda1", type=float, default=1.0, help="weight on L_dist (Eq. 19)")
    p.add_argument("--lambda2", type=float, default=2e-4, help="weight on R(K_eff) (Eq. 20)")
    p.add_argument("--lambda-warmup-epochs", type=float, default=30.0,
                    help="linearly ramp lambda1/lambda2 from 0 over this many epochs")

    # Assignment temperature schedule (Eq. 9-10, Section 0.2.2)
    p.add_argument("--tau-start", type=float, default=4.0)
    p.add_argument("--tau-end", type=float, default=60.0)
    p.add_argument("--tau-warmup-frac", type=float, default=0.1)

    # Dictionary / usage-gate hyperparameters (Sections 0.2.1, 0.2.3)
    p.add_argument("--K-mult", type=float, default=1.0,
                    help="initial over-complete dictionary size K = round(K_mult * C_out) per layer")
    p.add_argument("--K-min", type=int, default=4)
    p.add_argument("--gate-init-keep-prob", type=float, default=1.0)
    p.add_argument("--compile-threshold", type=float, default=0.01,
                    help="usage threshold on hardened z_k used both for logging K_eff and compilation")

    p.add_argument("--no-track-distortion", action="store_true",
                    help="disable the extra per-layer W-branch forward (ablation only; "
                         "removes L_dist and reduces Eq. 21 to L_task + lambda2*R(K_eff))")

    p.add_argument("--eval-freq", type=int, default=1, help="run val evaluation every N epochs")
    p.add_argument("--compiled-eval-freq", type=int, default=10,
                    help="also compile the model and evaluate the *compiled* network every N epochs "
                         "(always included on the final epoch regardless of this setting). This is the "
                         "number that actually matters for deployment -- it can diverge substantially "
                         "from the soft/uncompiled val accuracy above if usage gates haven't saturated "
                         "to near-binary values yet, so it's worth tracking over the whole run rather "
                         "than only reading it once at the end.")
    p.add_argument("--out-dir", type=str, default=None,
                    help="defaults to ./runs/<dataset>_<arch>_proto if omitted")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--no-plots", action="store_true", help="skip saving history.csv / accuracy_curve.png / loss_curve.png")
    args = p.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    if args.data_root is None:
        args.data_root = defaults["data_root"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.out_dir is None:
        args.out_dir = f"./runs/{args.dataset}_{args.arch}_proto"
    args.num_classes = defaults["num_classes"]
    args.input_size = defaults["input_size"]
    return args


def build_param_groups(model, weight_decay):
    """Weight decay on conv/fc kernels only; no decay on biases, BatchNorm
    affine params, the prototype dictionary P_raw, or the usage-gate logits
    eta (these are not part of the classical "weight magnitude" the decay
    term is meant to regularize, and P_raw is re-normalized every forward
    pass regardless)."""
    decay, no_decay = [], []
    for m in model.modules():
        if isinstance(m, PrototypeConv2d):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
            no_decay.append(m.P_raw)
            no_decay.append(m.gate.eta)
        elif isinstance(m, nn.Conv2d):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            no_decay.append(m.weight)
            no_decay.append(m.bias)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_one_epoch(model, loader, optimizer, device, epoch, args, global_step, total_steps):
    """Runs one training epoch and returns (new_global_step, stats) where
    `stats` has the epoch-averaged train loss/accuracy plus the schedule
    values (tau, lambda1, lambda2) as of the epoch's last step -- no
    intra-epoch printing; the caller prints one summary line per epoch."""
    model.train()
    task_meter, dist_meter, card_meter, acc_meter = (AverageMeter() for _ in range(4))
    t0 = time.time()
    tau = lam1 = lam2 = None

    for i, (x, y) in enumerate(loader):
        step = global_step + i
        tau = anneal_tau(step, total_steps, args.tau_start, args.tau_end, args.tau_warmup_frac)
        set_model_tau(model, tau)

        warmup_steps = int(args.lambda_warmup_epochs * len(loader))
        lam1 = linear_warmup(step, warmup_steps, args.lambda1)
        lam2 = linear_warmup(step, warmup_steps, args.lambda2)

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = model(x)  # uses W_hat at every layer (Section 0.3.2)
        task_loss = nn.functional.cross_entropy(logits, y)
        dist_term = distortion_loss(model) if not args.no_track_distortion else torch.zeros((), device=device)
        card_term = cardinality_reg(model)

        loss = task_loss + lam1 * dist_term + lam2 * card_term

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc1 = accuracy(logits.detach(), y, topk=(1,))[0]
        task_meter.update(task_loss.item(), x.size(0))
        dist_meter.update(float(dist_term.detach()), x.size(0))
        card_meter.update(float(card_term.detach()), x.size(0))
        acc_meter.update(acc1, x.size(0))

    stats = dict(
        train_loss=task_meter.avg, train_acc=acc_meter.avg,
        dist_loss=dist_meter.avg, card_reg=card_meter.avg,
        tau=tau, lambda1=lam1, lambda2=lam2, epoch_time=time.time() - t0,
    )
    return global_step + len(loader), stats


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (accuracy, mean cross-entropy loss) over `loader`."""
    model.eval()
    acc_meter, loss_meter = AverageMeter(), AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc1 = accuracy(logits, y, topk=(1,))[0]
        acc_meter.update(acc1, x.size(0))
        loss_meter.update(loss.item(), x.size(0))
    return acc_meter.avg, loss_meter.avg


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"dataset={args.dataset} arch={args.arch} num_classes={args.num_classes} "
          f"input_size={args.input_size} data_root={args.data_root}")

    train_loader, test_loader = build_dataset(
        args.dataset, args.data_root, args.batch_size, args.workers
    )

    proto_kwargs = dict(
        K_mult=args.K_mult,
        K_min=args.K_min,
        tau_init=args.tau_start,
        gate_init_keep_prob=args.gate_init_keep_prob,
        track_distortion=not args.no_track_distortion,
    )
    model = build_model(args.arch, args.num_classes, proto_kwargs).to(device)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
        best_ckpt = os.path.join(args.out_dir, "best.pt")
        best_acc = torch.load(best_ckpt, map_location=device)["acc"] if os.path.isfile(best_ckpt) else 0.0

    param_groups = build_param_groups(model, args.weight_decay)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, nesterov=True)
    if start_epoch > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", args.lr)
   # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs, last_epoch=start_epoch - 1 if start_epoch > 0 else -1)
    
    total_steps = args.epochs * len(train_loader)
    global_step = start_epoch * len(train_loader)
    best_acc = 0.0
    history = []

    # NOTE: CIFAR-10/100 (and this codebase's Tiny-ImageNet loader) only
    # provide a train/test split -- there is no separate validation set
    # carved out of the training data. What's printed and plotted as "val"
    # below is this held-out test split evaluated after every epoch (the
    # standard practice for these benchmarks); the "final test accuracy"
    # printed at the end is the same split, evaluated once more on the best
    # checkpoint -- it is not a third, previously-unseen split.
    for epoch in range(start_epoch, args.epochs):
        global_step, tr = train_one_epoch(model, train_loader, optimizer, device, epoch, args, global_step, total_steps)
        scheduler.step()

        val_acc, val_loss = evaluate(model, test_loader, device)
        report, k_eff_total, c_total = effective_cardinality_report(model, threshold=args.compile_threshold)

        do_compiled = ((epoch + 1) % args.compiled_eval_freq == 0) or (epoch == args.epochs - 1)
        compiled_acc = None
        if do_compiled:
            compiled = compile_model(copy.deepcopy(model).to(device).eval(), threshold=args.compile_threshold)
            compiled_acc, _ = evaluate(compiled, test_loader, device)
            del compiled

        line = (f"[epoch {epoch}] train_loss={tr['train_loss']:.4f} train_acc={tr['train_acc']:.2f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}  "
                f"L_dist={tr['dist_loss']:.4f} K_eff={k_eff_total}/{c_total} "
                f"({100.0*k_eff_total/max(1,c_total):.1f}%) tau={tr['tau']:.2f} "
                f"lam1={tr['lambda1']:.3g} lam2={tr['lambda2']:.3g} [{tr['epoch_time']:.1f}s]")
        if compiled_acc is not None:
            line += f"  compiled_acc={compiled_acc:.2f}"
        print(line)

        history.append(dict(
            epoch=epoch, train_loss=tr["train_loss"], train_acc=tr["train_acc"],
            val_loss=val_loss, val_acc=val_acc, dist_loss=tr["dist_loss"],
            card_reg=tr["card_reg"], k_eff_total=k_eff_total, c_total=c_total,
            tau=tr["tau"], lambda1=tr["lambda1"], lambda2=tr["lambda2"],
            compiled_acc=compiled_acc,
        ))

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        ckpt = {"model": model.state_dict(), "epoch": epoch + 1, "acc": val_acc, "args": vars(args)}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

    print(f"\nTraining complete. Best val acc1={best_acc:.2f}")

    # --- Compile & report the actual deployable network (Section 0.4) ---
    compiled = compile_model(copy.deepcopy(model).to(device).eval(), threshold=args.compile_threshold)
    compiled_acc, compiled_loss = evaluate(compiled, test_loader, device)
    final_test_acc, final_test_loss = evaluate(model, test_loader, device)

    input_size = (1, 3, args.input_size, args.input_size)
    ref_model = build_model(args.arch, args.num_classes, dict(use_proto=False)).to(device)
    dense_flops, dense_params = profile_model(ref_model, input_size=input_size, device=device)
    compiled_flops, compiled_params = profile_model(compiled, input_size=input_size, device=device)

    print("\n=== Final test-set results (same held-out split used as 'val' during training) ===")
    print(f"Uncompiled (final-epoch) network: test_acc={final_test_acc:.2f}  test_loss={final_test_loss:.4f}")
    print(f"Compiled network:                 test_acc={compiled_acc:.2f}  test_loss={compiled_loss:.4f}")
    print(f"Params: {compiled_params:,} / {dense_params:,} "
          f"({100.0*compiled_params/dense_params:.1f}% of a standard dense {args.arch})")
    print(f"MACs:   {compiled_flops:,} / {dense_flops:,} "
          f"({100.0*compiled_flops/dense_flops:.1f}% of a standard dense {args.arch})")

    torch.save(
        {"model": compiled.state_dict(),
         "compiled_acc": compiled_acc, "compiled_params": compiled_params, "compiled_flops": compiled_flops},
        os.path.join(args.out_dir, "compiled_summary.pt"),
    )

    if not args.no_plots:
        from plot_utils import plot_curves, save_history_csv
        csv_path = save_history_csv(history, args.out_dir)
        acc_path, loss_path = plot_curves(history, args.out_dir)
        print(f"\nSaved training history to {csv_path}")
        if acc_path:
            print(f"Saved accuracy_curve.png / loss_curve.png to {args.out_dir}")


if __name__ == "__main__":
    main()
