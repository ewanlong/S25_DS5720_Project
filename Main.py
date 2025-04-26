import torch as t
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import TransGNN
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import setproctitle
import itertools
import json
from datetime import datetime
import sys

class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        metrics = ['Loss', 'preLoss', 'Recall@20', 'NDCG@20', 'Recall@40', 'NDCG@40']
        for metric in metrics:
            self.metrics['Train' + metric] = list()
            self.metrics['Test' + metric] = list()
        self.best_metrics = {
            'Recall@20': 0,
            'NDCG@20': 0,
            'Recall@40': 0,
            'NDCG@40': 0
        }
        self.patience = args.patience
        self.patience_counter = 0
        self.early_stop = False

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            temp = name + metric
            if save and temp in self.metrics:
                self.metrics[temp].append(val)
        return ret[:-2] + '  '

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model is not None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, tstFlag))
                if self.early_stop:
                    log(f"Early stopping triggered: no improvement for {self.patience} epochs")
                    break
            print()
        if not self.early_stop:
            reses = self.testEpoch()
            log(self.makePrint('Test', args.epoch, reses, True))

    def prepareModel(self):
        self.model = TransGNN().to(device)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        for i, batch in enumerate(trnLoader):
            ancs, poss, negs = batch
            ancs = ancs.long().to(device)
            poss = poss.long().to(device)
            negs = negs.long().to(device)
            bprLoss = self.model.calcLosses(ancs, poss, negs, self.handler.torchBiAdj)
            loss = bprLoss
            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('Step %d/%d: loss = %.3f         ' % (i, steps, loss), save=False, oneline=True)
        return {'Loss': epLoss / steps, 'preLoss': epPreLoss / steps}

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall20, epNdcg20, epRecall40, epNdcg40 = [0] * 4
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for i, (usr, trnMask) in enumerate(tstLoader, 1):
            usr = usr.long().to(device)
            trnMask = trnMask.to(device)
            usrEmbeds, itmEmbeds = self.model.predict(self.handler.torchBiAdj)
            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = t.topk(allPreds, 40)
            recall20, ndcg20 = self.calcRes(topLocs[:, :20].cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            recall40, ndcg40 = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall20 += recall20
            epNdcg20 += ndcg20
            epRecall40 += recall40
            epNdcg40 += ndcg40
            log('Steps %d/%d: R@20 = %.2f, N@20 = %.2f, R@40 = %.2f, N@40 = %.2f          ' %
                (i, steps, recall20, ndcg20, recall40, ndcg40), save=False, oneline=True)
        
        ret = {
            'Recall@20': epRecall20 / num,
            'NDCG@20': epNdcg20 / num,
            'Recall@40': epRecall40 / num,
            'NDCG@40': epNdcg40 / num
        }

        all_improved = all(ret[metric] >= self.best_metrics[metric] for metric in self.best_metrics)
        any_improved = any(ret[metric] > self.best_metrics[metric] for metric in self.best_metrics)
        if all_improved and any_improved:
            for metric in self.best_metrics:
                self.best_metrics[metric] = max(self.best_metrics[metric], ret[metric])
            self.patience_counter = 0
            log("New best model found, all metrics non-decreasing, at least one improved")
            log(f"R@20={ret['Recall@20']:.4f}, N@20={ret['NDCG@20']:.4f}, R@40={ret['Recall@40']:.4f}, N@40={ret['NDCG@40']:.4f}")
            self.saveHistory()
            log("Best model saved")
        else:
            self.patience_counter += 1
            log(f"No improvement for {self.patience_counter} epochs. Current best metrics:")
            log(f"R@20={self.best_metrics['Recall@20']:.4f}, N@20={self.best_metrics['NDCG@20']:.4f}, R@40={self.best_metrics['Recall@40']:.4f}, N@40={self.best_metrics['NDCG@40']:.4f}")
            if self.patience_counter >= self.patience:
                self.early_stop = True
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            tempTopLocs = list(topLocs[i])
            tempTstLocs = tstLocs[batIds[i]]
            tstNum = len(tempTstLocs)
            k = len(tempTopLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, k))])
            recall = dcg = 0
            for val in tempTstLocs:
                if val in tempTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(tempTopLocs.index(val) + 2))
            recall /= tstNum
            ndcg = dcg / maxDcg if maxDcg > 0 else 0
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        history_dir = '../History/'
        os.makedirs(history_dir, exist_ok=True)
        with open(history_dir + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)
        epoch_metrics = {metric_name: values for metric_name, values in self.metrics.items()}
        with open(history_dir + args.save_path + '_epochs.json', 'w') as f:
            json.dump({
                'metrics': epoch_metrics,
                'best_metrics': self.best_metrics,
                'params': {
                    'lr': args.lr,
                    'decay': args.decay,
                    'att_head': args.att_head,
                    'edgeSampRate': args.edgeSampRate,
                    'dropout': args.dropout,
                    'num_head': args.num_head,
                    'trans_layer': args.trans_layer
                }
            }, f, indent=4)
        models_dir = '../Models/'
        os.makedirs(models_dir, exist_ok=True)
        t.save({'model': self.model, 'params': args.__dict__}, models_dir + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        models_dir = '../Models/'
        os.makedirs(models_dir, exist_ok=True)
        history_dir = '../History/'
        os.makedirs(history_dir, exist_ok=True)
        checkpoint = t.load(models_dir + args.load_model + '.mod')
        self.model = checkpoint['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        with open(history_dir + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')




def run_with_params(params_dict, run_id=0):
    original_params = {}
    for param_name, param_value in params_dict.items():
        original_params[param_name] = getattr(args, param_name)
        setattr(args, param_name, param_value)
    param_str = '_'.join([f"{k}_{v}" for k, v in params_dict.items()])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_save_path = args.save_path
    original_load_model = args.load_model
    args.save_path = f"{original_save_path}_{param_str}_{timestamp}_{run_id}"
    args.load_model = None

    params_dir = os.path.abspath('../Params_Records/')
    os.makedirs(params_dir, exist_ok=True)
    params_file = os.path.join(params_dir, f"{args.save_path}_params.json")
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=4)
    log(f'Using parameters: {params_dict}, Run ID: {run_id}, Save path: {args.save_path}')
    
    handler = DataHandler()
    handler.LoadData()
    coach = Coach(handler)
    coach.run()
    
    results = coach.best_metrics.copy()
    
    for param_name, param_value in original_params.items():
        setattr(args, param_name, param_value)
    args.save_path = original_save_path
    args.load_model = original_load_model
    
    return results

def param_search():
    log('Starting parameter search')
    search_space = {
        'lr': [1e-4, 5e-4, 1e-3, 5e-3],
        'decay': [0, 1e-5, 1e-4, 1e-3, 1e-2],
        'att_head': [2, 4, 8, 16, 32],
        'edgeSampRate': [5, 10, 15, 20, 25, 30, 35],
        'dropout': [0, 0.1, 0.2, 0.3, 0.5],
        'num_head': [2, 4, 8, 16, 32],
        'trans_layer': [1, 2, 3, 4]
    }
    search_dir = '../Search_Results/'
    os.makedirs(search_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_result_file = f"{search_dir}search_results_{timestamp}.json"
    all_search_results = []
    log_dir = '../Logs/'
    os.makedirs(log_dir, exist_ok=True)

    all_param_combinations = []
    for lr in search_space['lr']:
        for decay in search_space['decay']:
            for att_head in search_space['att_head']:
                for edgeSampRate in search_space['edgeSampRate']:
                    for dropout in search_space['dropout']:
                        for num_head in search_space['num_head']:
                            for trans_layer in search_space['trans_layer']:
                                params = {
                                    'lr': lr,
                                    'decay': decay,
                                    'att_head': att_head,
                                    'edgeSampRate': edgeSampRate,
                                    'dropout': dropout,
                                    'num_head': num_head,
                                    'trans_layer': trans_layer
                                }
                                all_param_combinations.append(params)
    log(f'Generated {len(all_param_combinations)} parameter combinations')

    for i, params in enumerate(all_param_combinations):
        import hashlib
        param_hash = hashlib.md5(str(params).encode()).hexdigest()[:8]
        param_str = f"params_{param_hash}"
        param_result_file = f"{search_dir}search_{param_str}.json"
        
        param_detail_file = f"{search_dir}detail_{param_str}.json"
        if not os.path.exists(param_detail_file):
            with open(param_detail_file, 'w') as f:
                json.dump(params, f, indent=4)
        
        if os.path.exists(param_result_file):
            log(f"Parameter set {param_hash} already trained. Skipping.")
            with open(param_result_file, 'r') as f:
                combo_result = json.load(f)
            all_search_results.append(combo_result)
            continue

        param_log_file = f"{log_dir}param_search_{param_str}_{timestamp}.txt"
        logger.logFilePath = param_log_file
        log(f'Running parameter combination {i+1}/{len(all_param_combinations)}: {param_str}')

        run_results = []
        for run in range(args.search_runs):
            run_log_file = f"{log_dir}param_{param_str}_run_{run}_{timestamp}.txt"
            logger.logFilePath = run_log_file
            log(f'Running run {run+1}/{args.search_runs} for combination {param_str}')
            results = run_with_params(params, run)
            run_results.append(results)
            run_result_file = f"{search_dir}{param_str}_run_{run}.json"
            with open(run_result_file, 'w') as f:
                json.dump(results, f, indent=4)

        avg_results = {'params': params}
        for metric in ['Recall@20', 'NDCG@20', 'Recall@40', 'NDCG@40']:
            avg_results[metric] = np.mean([res[metric] for res in run_results])
        all_search_results.append(avg_results)
        log(f'Params: {params}, Avg results: R@20={avg_results["Recall@20"]:.4f}, N@20={avg_results["NDCG@20"]:.4f}, R@40={avg_results["Recall@40"]:.4f}, N@40={avg_results["NDCG@40"]:.4f}')
        
        with open(param_result_file, 'w') as f:
            json.dump(avg_results, f, indent=4)
        with open(overall_result_file, 'w') as f:
            json.dump({
                'all_results': all_search_results,
                'progress': f'{i+1}/{len(all_param_combinations)}'
            }, f, indent=4)

    best_result = max(all_search_results, key=lambda x: (x['Recall@20'] + x['NDCG@20'] + x['Recall@40'] + x['NDCG@40'])/4)
    best_params = best_result['params']
    with open(overall_result_file, 'w') as f:
        json.dump({
            'all_results': all_search_results,
            'best_params': best_params,
            'best_metrics': {
                'Recall@20': best_result['Recall@20'],
                'NDCG@20': best_result['NDCG@20'],
                'Recall@40': best_result['Recall@40'],
                'NDCG@40': best_result['NDCG@40']
            }
        }, f, indent=4)
    log(f'Parameter search completed, best parameters: {best_params}')
    log(f'Best metrics: R@20={best_result["Recall@20"]:.4f}, N@20={best_result["NDCG@20"]:.4f}, R@40={best_result["Recall@40"]:.4f}, N@40={best_result["NDCG@40"]:.4f}')
    return best_params

def generate_background_command():
    """Generate Windows background run command."""
    script_path = os.path.abspath(__file__)
    cmd = f'start /B python "{script_path}" --background True > NUL'
    print(f"Windows background run command: {cmd}")
    return cmd

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if args.background:
        log_dir = os.path.join(base_dir, 'Logs')
        os.makedirs(log_dir, exist_ok=True)
        background_log = os.path.join(log_dir, f'background_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        log_file = open(background_log, 'w', encoding='utf-8')
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"=== Background run started at {datetime.now()} ===")
        print(f"Command line args: {sys.argv}")
        print(f"CUDA available: {t.cuda.is_available()}")
        if t.cuda.is_available():
            print(f"GPU used: {t.cuda.get_device_name(0)}")
        import atexit
        def cleanup():
            if not log_file.closed:
                print(f"=== Background run ended at {datetime.now()} ===")
                log_file.close()
        atexit.register(cleanup)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = '../Logs/'
    os.makedirs(log_dir, exist_ok=True)
    main_log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    with open(main_log_file, 'w') as f:
        f.write("Log file created\n")
    print(f"Log file created: {main_log_file}")
    
    logger.logFilePath = main_log_file
    logger.saveDefault = True
    
    print(f"CUDA available: {t.cuda.is_available()}")
    if t.cuda.is_available():
        print(f"Using GPU: {t.cuda.get_device_name(0)}")
        log(f"Using GPU: {t.cuda.get_device_name(0)}")
    else:
        log("Training on CPU")

    original_load_model = args.load_model

    if args.do_search:
        args.load_model = None
        search_log_file = f"{log_dir}param_search_{timestamp}.txt"
        logger.logFilePath = search_log_file
        log(f"Parameter search logs saved to: {search_log_file}")
        best_params = param_search()
        for param_name, param_value in best_params.items():
            setattr(args, param_name, param_value)
        log(f'Using best parameters: {best_params}')

    logger.logFilePath = main_log_file
    all_results = {'Recall@20': [], 'NDCG@20': [], 'Recall@40': [], 'NDCG@40': []}

    for run in range(args.num_runs):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_file = f"{log_dir}run_{run+1}_{run_timestamp}.txt"
        logger.logFilePath = run_log_file
        log(f'Starting run {run+1}/{args.num_runs}, log saved to: {run_log_file}')
        
        original_save_path = args.save_path
        args.save_path = f"{original_save_path}_{run_timestamp}_{run+1}"
        args.load_model = None
        
        handler = DataHandler()
        handler.LoadData()
        coach = Coach(handler)
        coach.run()
        
        results = coach.best_metrics.copy()
        for metric in all_results:
            all_results[metric].append(results[metric])
        
        log(f'Run {run+1} completed')
        log(f'R@20={results["Recall@20"]:.4f}, N@20={results["NDCG@20"]:.4f}, R@40={results["Recall@40"]:.4f}, N@40={results["NDCG@40"]:.4f}')

    logger.logFilePath = main_log_file
    args.load_model = original_load_model

    avg_results = {metric: np.mean(all_results[metric]) for metric in all_results}
    log('Average results from all runs:')
    log(f'Avg R@20={avg_results["Recall@20"]:.4f}, Avg N@20={avg_results["NDCG@20"]:.4f}')
    log(f'Avg R@40={avg_results["Recall@40"]:.4f}, Avg N@40={avg_results["NDCG@40"]:.4f}')

    results_dir = '../Results/'
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + original_save_path + f'_all_runs_{timestamp}.pkl', 'wb') as f:
        pickle.dump({
            'individual_results': all_results,
            'average_results': avg_results,
            'params': args.__dict__
        }, f)
    log(f'All results saved to {results_dir + original_save_path}_all_runs_{timestamp}.pkl')
    log(f'Training completed, main log saved to: {main_log_file}')

if args.generate_cmd:
    generate_background_command()
    sys.exit(0)
