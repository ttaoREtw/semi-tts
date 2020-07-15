import os
import sys
import abc
import math
import torch
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from src.util import Timer, human_format

TB_FLUSH_FREQ = 180

class BaseSolver():
    ''' 
    Prototype Solver for all kinds of tasks
        config - yaml-styled config
        paras  - argparse outcome
    '''
    def __init__(self,config,paras,mode):
        # General Settings
        self._GRAD_CLIP = 5.0
        self._PROGRESS_STEP = 20       # Std. output refresh freq.
        self._DEV_N_EXAMPLE = 4        # Number of examples (alignment/text) to show in tensorboard

        self.config = config
        self.paras = paras
        self.device = torch.device('cuda') if self.paras.gpu and torch.cuda.is_available() else torch.device('cpu')

        # Name experiment
        self.exp_name = paras.name
        if self.exp_name is None:
            self.exp_name = '-'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
        
        # Filepath setup
        os.makedirs(paras.ckpdir, exist_ok=True)
        self.ckpdir = os.path.join(paras.ckpdir,self.exp_name)
        os.makedirs(self.ckpdir, exist_ok=True)

        self.logdir = os.path.join(paras.logdir,self.exp_name)
        # Training attributes
        if mode == 'train':
            # Logger settings
            self.log = SummaryWriter(self.logdir, flush_secs = TB_FLUSH_FREQ)
            self.timer = Timer()
            # Hyperparameters
            self.step = 0
            self.valid_step = config['hparas']['valid_step']
            self.max_step = config['hparas']['max_step']

    # ------------------------ Abstract methods, implement these ----------------------------- #

    @abc.abstractmethod
    def load_data(self):
        '''
        Called by main to load all data
        After this call, data related attributes should be setup (e.g. self.tr_set, self.dev_set)
        The following MUST be setup for training
            -
            - self.tr_set (torch.utils.data.Dataloader)
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def set_model(self):
        '''
        Called by main to set models
        After this call, model related attributes should be setup (e.g. self.l2_loss)
        The followings MUST be setup
            - self.model (torch.nn.Module)
            - self.optimizer (src.Optimizer),
                init. w/ self.optimizer = src.Optimizer(self.model.parameters(),**self.config['hparas'])
        Loading pre-trained model should also be performed here 
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def exec(self):
        '''
        Called by main to execute training/inference
        
        ############# For training, copy and modify the following #############
        self.verbose(['Total training steps {}.'.format(human_format(self.max_step))])
        self.timer.set()

        while self.step<self.max_step:
            for data in self.tr_set:
                # pre-step
                _ = self.optimizer.pre_step(self.step)       # Catch the returned tf_rate if needed
                data,of,your,format = self.fetch_data(data)
                self.timer.cnt('rd')

                # forward
                your, model, output = self.model(data,of,your,format)
                loss = compute_your_loss(...)
                self.timer.cnt('fw')

                # backward
                grad_norm = self.backward(loss)

                # Log
                if self.step%self._PROGRESS_STEP==0:
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'\
                            .format(loss.cpu().item(),grad_norm,self.timer.show()))
                    self.write_log('loss',{'tr':loss})
                    self.write_log('some_other',{things:to_log})

                # Validation
                if self.step%self.valid_step == 0:
                    self.validate() # check end of this file for example

                # End of step
                self.step+=1
                self.timer.set()
                if self.step > self.max_step:break
        ###########################################

        '''
        raise NotImplementedError

    @abc.abstractclassmethod
    def fetch_data(self, data):
        ''' 
        input `data` should be whatever Dataloader returns
        Parse the data, do smt., move to self.device and return
        Quick example:
            audio = data[0].to(self.device)
            txt   = data[1].to(self.device)
            txt_len =  torch.sum(txt!=0,dim=-1)
            return audio, txt, txt_len
        '''
        raise NotImplementedError


    # ------------------------ Default methods ------------------------------- #

    def backward(self, loss):
        '''
        Standard backward step with self.timer and debugger
            <torch> loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._GRAD_CLIP)
        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.step))
        else:
            self.optimizer.step()
        self.timer.cnt('bw')
        return grad_norm

    def verbose(self,msg):
        ''' 
        Verbose function for print information to stdout
            <list> msg - List of <str> for verbosity
        '''
        if self.paras.verbose:
            self._clean_line()
            if type(msg) is str:
                print('[INFO]',msg)
            else:
                for m in msg:
                    print('[INFO]',m)

    def progress(self,msg):
        '''
        Verbose function for updating progress on stdout
            <str> msg - String to show as progress bar, do not include "new line"
        '''
        if self.paras.verbose:
            self._clean_line()
            print('[{}] {}'.format(human_format(self.step),msg),end='\r')

    def _clean_line(self):
        sys.stdout.write("\033[K")
    
    def write_log(self,log_name,log_value):
        '''
        Write log to TensorBoard
            <str> log_name           - Name of tensorboard variable
            <dict>/<array> log_value - Value of variable (e.g. dict of losses, spectrogram, ..), passed if value = None
        '''
        if type(log_value) is dict:
            log_value = {key:val for key, val in log_value.items() if (val is not None and not math.isnan(val))}

        if log_value is None:
            pass
        elif len(log_value)>0:
            # ToDo : support all types of input
            if 'align' in log_name or 'spec' in log_name or 'hist' in log_name:
                img, form = log_value
                self.log.add_image(log_name,img, global_step=self.step, dataformats=form)
            elif 'code' in log_name:
                self.log.add_embedding(log_value[0], metadata=log_value[1], tag=log_name, global_step=self.step)
            elif 'wave' in log_name:
                signal, sr = log_value
                self.log.add_audio(log_name, torch.FloatTensor(signal).unsqueeze(0), self.step, sr)
            elif 'text' in log_name or 'hyp' in log_name:
                self.log.add_text(log_name, log_value, self.step)
            else:
                self.log.add_scalars(log_name,log_value,self.step)

    def save_checkpoint(self, f_name, score):
        '''' 
        Ckpt saver
            <str> f_name  - the name phnof ckpt file (w/o prefix) to store, overwrite if existed
            <float> score - The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            "global_step": self.step,
        }
        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (step = {}, score = {:.2f}) and status @ {}".\
                                       format(human_format(self.step),score,ckpt_path))


    '''
    ############# Example of validation for training, copy and modify the following #############
    def validate(self):
        # Eval mode
        self.model.eval()
        dev_loss = []

        for i,data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1,len(self.dv_set)))
            # Fetch data
            data1, data2 = self.fetch_data(data)

            # Forward model
            with torch.no_grad(): 
                pred, _ = self.model(data1)
            loss = cal_loss(pred,data2)
            dev_loss.append(loss)
        
        # Ckpt if performance improves (replace self.best_loss with your metric)
        dev_loss = sum(dev_loss)/len(dev_loss)
        if dev_loss < self.best_loss :
            self.best_loss = dev_loss
            self.save_checkpoint('best_loss.pth',dev_loss)
        self.write_log('loss',{'dv':dev_loss})

        # Show some example of last batch on tensorboard
        for i in range(min(len(txt),self._DEV_N_EXAMPLE)):
            if self.step ==0:
                self.write_log('true_text{}'.format(i),tensor2txt(data2[i]))
            self.write_log('pred_text{}'.format(i),tensor2txt(pred[i]))

        # Resume training
        self.model.train()
    ###############################################################################################
    '''
