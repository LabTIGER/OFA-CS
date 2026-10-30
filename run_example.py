
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import json
from model import PretrainModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as TorData
# from accuracy_globalsearch import epoch_evaluate 
from aa_utils import *



if __name__ == "__main__":

    args = parse_args()
    print(args)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # device
    device = torch.device(args.device)
    # args.device = device

    # === 指定多个训练/测试数据集（在此处设置） ===
    datasets_train = ['photo']   # 示例：['dblp','photo'] or ['citeseer']
    datasets_test = ['citeseer']        # 示例：['cora']'texas','cornell','wisconsin'

    # dataset creation params
    dims = 64
    # 构建 Dataset 对象列表（加载图数据）
    data_train_list = [Dataset(dims, name) for name in datasets_train]
    data_test_list = [Dataset(dims, name) for name in datasets_test]

 
    train_prepared = []
    for dt in data_train_list:
        features = dt.graph.x  # [N, dims]
        adj = dt.graph.adj.to(device)
        label = dt.graph.y  
        local_token_seqs = dt.graph.local_token_seqs  # [N, hops+1, local_seq_len]
        global_token_seqs = dt.graph.global_token_seqs  # [N, hops+1, global_seq_len]
        pos_samples = dt.graph.pos_samples  # [N, pos_num]
        neg_samples = dt.graph.neg_samples  # [N, pos_num]
        adj = transform_coo_to_csr(adj)
        adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0])
        dataset = TorData.TensorDataset(local_token_seqs, global_token_seqs)
        data_loader = TorData.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        train_prepared.append({
            "name": dt.name if hasattr(dt, 'name') else None,
            "features": features,
            "adj_csr": adj,
            "adj_batch": adj_batch,
            "minus_adj_batch": minus_adj_batch,
            "global_token_seqs": global_token_seqs,
            "local_token_seqs": local_token_seqs,
            "pos_samples": pos_samples,
            "neg_samples": neg_samples,
            "data_loader": data_loader,
            "num_nodes": features.shape[0]
        })

    # Precompute for test datasets (we need tokenized_seqs and few_shot_com for test_forward)
    test_prepared = []
    for dt in data_test_list:
        test_prepared.append({
            "name": dt.name if hasattr(dt, 'name') else None,
            "global_token_seqs": dt.graph.global_token_seqs,
            "local_token_seqs": dt.graph.local_token_seqs,
            "num_nodes": dt.graph.x.shape[0]
        })


    model = PretrainModel(input_dim=dims, config=args).to(device)

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
                    optimizer,
                    warmup_updates=args.warmup_updates,
                    tot_updates=args.tot_updates,
                    lr=args.peak_lr,
                    end_lr=args.end_lr,
                    power=1.0)


    print("starting training...")
    model.train()

    bce_loss = torch.nn.BCELoss()
    loss_train_b = []
    os.makedirs(args.save_path, exist_ok=True)
    ckpt_path = os.path.join(args.save_path, "last_epoch.pt")
    early_stopper = LossEarlyStopper(patience=5, ckpt_path=ckpt_path)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_step_cnt = 0
        # iterate through all training datasets
        for ds_idx, ds_info in enumerate(train_prepared):
            data_loader = ds_info["data_loader"]
            adj_batch_list = ds_info["adj_batch"]
            minus_adj_batch_list = ds_info["minus_adj_batch"]
            pos_samples = ds_info["pos_samples"]
            neg_samples = ds_info["neg_samples"]
            
            for index, (local_batch, global_batch) in enumerate(data_loader):
                local_sequence = local_batch.to(device)
                global_sequence = global_batch.to(device)
                # adj batches might be shorter/longer; guard with modulo if necessary, but ideally lengths match
                adj_ = adj_batch_list[index].to(device)
                minus_adj = minus_adj_batch_list[index].to(device)

                optimizer.zero_grad()
                node_tensor, neighbor_tensor, con_tensor, X_prime_dummy, noise_node, noise_pred_node, noise_neighbor, noise_pred_neighbor = model(local_sequence,global_sequence)

                loss_Trans = model.Transformer_loss(node_tensor, neighbor_tensor, pos_samples, neg_samples, adj_, minus_adj)

                loss_diff_node = F.mse_loss(noise_pred_node, noise_node)
                loss_diff_neighbor = F.mse_loss(noise_pred_neighbor, noise_neighbor)
                loss_diff = loss_diff_node + loss_diff_neighbor
                loss_train =  loss_diff + loss_Trans
                loss_train.backward()
                optimizer.step()
                lr_scheduler.step()

                loss_train_b.append(loss_train.item())

        # 直观地查看每个epoch在各测试数据集上的效果 (展示训练效果作用)
        model.eval()
        with torch.no_grad():
            for test_info in test_prepared:
                global_tokens = test_info["global_token_seqs"].to(device)
                local_tokens = test_info["local_token_seqs"].to(device)
                # model.test_forward 返回 X_prime
                X_prime = model.test_forward(local_tokens, global_tokens)
                # epoch_evaluate 打印/返回评价指标（根据你原来的实现）
                f1_val = epoch_evaluate(X_prime, test_info.get("name"))
                ds_name = test_info.get("name", "unknown")
                print(f"Validation - Dataset: {ds_name} | F1: {f1_val:.4f}")
        model.train()

        # if early_stopper.step(loss_train_b[-1], model, epoch=epoch+1):
        #     print("Early stopping triggered. Saving last epoch model and moving to final test.")
        #     break

        print('Epoch: {:04d} | last_batch_loss: {:.4f} | epoch_time: {:.2f}s'.format(
            epoch + 1, loss_train_b[-1] if len(loss_train_b) > 0 else 0.0, time.time() - epoch_start_time))

    print("Training finished.")

    results = []
    with torch.no_grad():
        for test_info in test_prepared:
            global_tokens = test_info["global_token_seqs"].to(device)
            local_tokens = test_info["local_token_seqs"].to(device)
            X_prime = model.test_forward(local_tokens, global_tokens)
            f1_val = epoch_evaluate(X_prime, test_info.get("name"))
            ds_name = test_info.get("name", "unknown")
            print(f"[Final Test] Dataset: {ds_name} | F1: {f1_val:.4f}")
            results.append((ds_name, float(f1_val.detach().cpu().item())))

    model_path = os.path.join(
        args.save_path,
        f"ofacs_photo_to_citeseer_seed{args.seed}_epoch{args.epochs}.pt",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_datasets": datasets_train,
            "target_datasets": datasets_test,
            "seed": args.seed,
            "epochs": args.epochs,
        },
        model_path,
    )

    summary_path = os.path.join(args.save_path, "photo_to_citeseer_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_datasets": datasets_train,
                "target_datasets": datasets_test,
                "seed": args.seed,
                "epochs": args.epochs,
                "final_f1": {name: value for name, value in results},
                "model_path": model_path,
            },
            f,
            indent=2,
        )

    print(f"Saved model to: {model_path}")
    print(f"Saved summary to: {summary_path}")







