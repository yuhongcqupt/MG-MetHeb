import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import argparse
import time
import util
from engine import trainer
import sys
import random

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--study_area', type=str, default='SF', help='study area')
parser.add_argument('--data', type=str, default='../data/',
                    help='data path')
parser.add_argument('--adjdata', type=str, default='../data/',
                    help='adj data path')
parser.add_argument('--save', type=str, default='../dyn_only/',
                    help='save path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_false', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_false', help='whether add adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=7, help='inputs dimension')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=30, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--fuse_init', type=str, default='1,1,1',
                    help='')


args = parser.parse_args()
fuse_init = [float(x) for x in args.fuse_init.split(',')]
s = sum(fuse_init)
fuse_init = [x / s for x in fuse_init]


def main(expid):
    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype, args.stduy_area)
    if args.study_area == 'Seoul':
        adj_mx = util.load_adj([args.adjdata1, args.adjdata2], args.adjtype, args.study_area)
    elif args.study_area == 'Seattle':
        adj_mx = util.load_adj(args.adjdata, args.adjtype, args.study_area)
    else:
        adj_mx = util.load_adj(args.adjdata, args.adjtype, args.study_area)
    dataloader, adp = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    adp = torch.Tensor(adp).to(device)
    print(adp.size())
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    print(args)

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,fuse_init=tuple(fuse_init))
    print(sum(p.numel() for p in engine.model.parameters() if p.requires_grad))
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    best_val = float('inf')
    best_epoch = -1
    best_weights = None  

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, batch  in enumerate(dataloader['train_loader'].get_iterator()):
            if len(batch) == 4:
                 x, xd, xw, y = batch
            elif len(batch) == 3:
                x, xd, y = batch
                xw = xd
            else:
                x, y = batch
                xd = x
                xw = x
            assert x.shape[-1] == args.in_dim, (x.shape, args.in_dim)
            assert xd.shape[-1] == args.in_dim, (xd.shape, args.in_dim)
            assert xw.shape[-1] == args.in_dim, (xw.shape, args.in_dim)

            trainx_r = torch.tensor(x,  device=device, dtype=torch.float32).transpose(1, 3)  # [B,C,N,12]
            trainx_d = torch.tensor(xd, device=device, dtype=torch.float32).transpose(1, 3)  # [B,C,N,12]
            trainx_w = torch.tensor(xw, device=device, dtype=torch.float32).transpose(1, 3)

            trainy   = torch.tensor(y,  device=device, dtype=torch.float32).transpose(1, 3)  # [B,1,N,12]
            y_full   = trainy[:, 0, :, :]    
            metrics = engine.train(trainx_r, trainx_d, trainx_w,y_full)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)


        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, batch in enumerate(dataloader['val_loader'].get_iterator()):
            #print(len(batch))
            if len(batch) == 4:
                 x, xd, xw, y = batch
            elif len(batch) == 3:
                x, xd, y = batch
                xw = xd
            else:
                x, y = batch
                xd = x
                xw = x
            
            valx_r = torch.tensor(x,  device=device, dtype=torch.float32).transpose(1, 3)
            valx_d = torch.tensor(xd, device=device, dtype=torch.float32).transpose(1, 3)
            valx_w = torch.tensor(xw, device=device, dtype=torch.float32).transpose(1, 3)

            valy   = torch.tensor(y,  device=device, dtype=torch.float32).transpose(1, 3)
            y_full = valy[:, 0, :, :]
            metrics = engine.eval(valx_r, valx_d, valx_w,y_full)

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)


        if mvalid_loss < best_val:
            best_val = mvalid_loss
            best_epoch = i  

            model_obj = engine.model.module if hasattr(engine.model, "module") else engine.model
            if hasattr(model_obj, 'fuse_weights'):
                best_weights = model_obj.fuse_weights
            else:
                
                if hasattr(model_obj, 'alpha'):
                    w = torch.softmax(model_obj.alpha, dim=0).detach().cpu().numpy()
                else:
                    w = torch.softmax(model_obj.weight_logits, dim=0).detach().cpu().numpy()
                best_weights = [float(wi) for wi in w]

           
            print(f"[BEST UPDATE] Epoch {best_epoch} | Weights -> recent: {best_weights[0]:.4f} | daily: {best_weights[1]:.4f}" + 
                (f" | weekly: {best_weights[2]:.4f}" if len(best_weights) >= 3 else ""))

           
            try:
                import json, os
                os.makedirs(os.path.join(args.save, 'best'), exist_ok=True)
                meta = {
                    "best_epoch": best_epoch,
                    "best_valid_loss": float(best_val),
                    "fuse_weights": best_weights,
                    "fuse_init": fuse_init
                }
                with open(os.path.join(args.save, 'best', f"exp_{expid}_best_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                print(f"{e}")


        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
            flush=True)



        model_obj = engine.model.module if hasattr(engine.model, "module") else engine.model

        if hasattr(model_obj, "df_weights_mean") and model_obj.df_weights_mean is not None:
            w_dyn = model_obj.df_weights_mean  # [3]
            print(f"[DF mean] recent: {w_dyn[0]:.4f} | daily: {w_dyn[1]:.4f} | weekly: {w_dyn[2]:.4f}")
        else:
           
            w = torch.softmax(model_obj.alpha, dim=0).detach().cpu().numpy()
            print(f"[STATIC] recent: {w[0]:.4f} | daily: {w[1]:.4f} | weekly: {w[2]:.4f}")

        w = torch.softmax(engine.model.alpha, dim=0).detach().cpu().numpy()
        print(f"recent: {w[0]:.4f} | daily: {w[1]:.4f} | weekly: {w[2]:.4f}")



        torch.save(engine.model.state_dict(),
                   args.save + "exp_" + str(expid) + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    
    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "exp_" + str(expid) + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    engine.model.eval()

    total_inference_time = 0
    total_samples = 0

    for iter, batch in enumerate(dataloader['test_loader'].get_iterator()):
        if len(batch) == 4:
                x, xd, xw, y = batch
        elif len(batch) == 3:
            x, xd, y = batch
            xw = xd
        else:
            x, y = batch
            xd = x
            xw = x
        testx_r = torch.tensor(x,  device=device, dtype=torch.float32).transpose(1, 3)
        testx_d = torch.tensor(xd, device=device, dtype=torch.float32).transpose(1, 3)
        testx_w = torch.tensor(xw, device=device, dtype=torch.float32).transpose(1, 3)
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            preds = engine.model(testx_r, testx_d, testx_w)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            total_samples += testx_r.size(0)
 
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze(1))  

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    avg_time = total_inference_time / total_samples

    print("Total inference time: {:.4f}s".format(total_inference_time))

    print(
        "Average inference time per sample: {:.4f}s ({:.2f} ms)"
        .format(avg_time, avg_time * 1000)
    )
    print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    horizons = realy.size(-1)
    for i in range(horizons):
        pred = yhat[:, :, i]
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    print('best epoch: ', str(bestid))
    torch.save(engine.model.state_dict(),
               args.save + 'best/exp_' + str(expid) + '_best_epoch_' + str(bestid) +
               '_' + str(round(his_loss[bestid], 2)) + ".pth")
    print(f"[BEST FINAL] Epoch {best_epoch} | Valid Loss {best_val:.4f}")
    if best_weights is not None:
        if len(best_weights) == 2:
            print(f"Best Weights -> recent: {best_weights[0]:.4f} | daily: {best_weights[1]:.4f}")
        else:
            print(f"Best Weights -> recent: {best_weights[0]:.4f} | daily: {best_weights[1]:.4f} | weekly: {best_weights[2]:.4f}")

if __name__ == "__main__":
    t1 = time.time()

    main(0)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
