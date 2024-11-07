from tqdm import tqdm
from util.metric import eval_depth
import random
import torch
import torch.nn.functional as F
import os
def train(model,args,trainloader,valloader,optimizer,criterion,device):
    total_iters = args.epochs * len(trainloader)
    best_rmse = 2
    for epoch in range(args.epochs):
        model.train()
        running_loss = []
        evaluate_list = []
        total_loss = 0
        for i, sample in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            images = sample['image'].to(device)
            depths = sample['depth'].to(device)
            valid_mask = sample['valid_mask'].to(device)

            optimizer.zero_grad()
            if random.random() < 0.5:
                images = images.flip(-1)
                depths = depths.flip(-1)
                valid_mask = valid_mask.flip(-1)
            outputs = model(images)
            loss = criterion(outputs, depths, (valid_mask == 1) & (depths >= args.min_depth) & (depths <= args.max_depth))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10.0
        print(f"Epoch [{epoch + 1}/{args.epochs}] completed. Average Loss: {total_loss / len(trainloader):.4f}")
        running_loss.append(total_loss)
        model.eval()
        results = {'d1':0, 'd2':0,'d3': 0, 'abs_rel':0,'sq_rel':0, 'rmse':0,'rmse_log':0, 'log10':0,'silog':0}
        nsamples = 0
        with torch.no_grad():
            for i, sample in enumerate(valloader):
                img = sample['image'].to(device).float()
                depth = sample['depth'].to(device)
                valid_mask = sample['valid_mask'].to(device)  
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[:,0]
                for b in range(pred.shape[0]):
                    mask = (valid_mask[b] == 1) & (depth[b] >= args.min_depth) & (depth[b] <= args.max_depth)
                    if mask.sum() < 10:
                        continue
                    cur_results = eval_depth(pred[b][mask], depth[b][mask])
                    for k in results.keys():
                        results[k] += cur_results[k]
                    nsamples += 1
        if nsamples > 0:
            for k in results.keys():
                results[k] /= nsamples
            print(', '.join([f'{k}: {v:.3f}' for k, v in results.items()]))
            evaluate_list.append(results)
        if results['rmse'] < best_rmse:
            best_rmse = results['rmse']
            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),'epoch': epoch,'previous_best': results,}
<<<<<<< HEAD
            torch.save(checkpoint, os.path.join(args.save_path, 'wavelet_best.pth'))
    return running_loss,evaluate_list
=======
            torch.save(checkpoint, os.path.join(args.save_path, 'latest_new.pth'))
    return running_loss
>>>>>>> 59c627f1407b74c01eaafdb52d9a9ce20e303770

def evaluate(model, valloader, device, min_depth=0.1, max_depth=10.0):
    model.eval()
    results = {}
    nsamples = 0 

    with torch.no_grad():
        for i, sample in enumerate(valloader):
            img = sample['image'].to(device).float()
            depth = sample['depth'].to(device)
            valid_mask = sample['valid_mask'].to(device)  

            pred = model(img)
            pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[:, 0]

            for b in range(pred.shape[0]):
                mask = (valid_mask[b] == 1) & (depth[b] >= min_depth) & (depth[b] <= max_depth)
                if mask.sum() < 10:
                    continue

                cur_results = eval_depth(pred[b][mask], depth[b][mask])

                if not results:
                    results = {k: 0.0 for k in cur_results.keys()}
                for k in results.keys():
                    results[k] += cur_results[k]
                nsamples += 1
        if nsamples > 0:
            for k in results.keys():
                results[k] /= nsamples
            print(', '.join([f'{k}: {v:.3f}' for k, v in results.items()]))
        else:
            print("No valid samples found for evaluation.")

    return results
