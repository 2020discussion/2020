import os, sys
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
"""
师姐使用pytorch的实验代码，刚好最近需要学习
"""


class Cnn(nn.Module):
    def __init__(self, in_dim, out):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 512, kernel_size=3, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(512, 1024,  kernel_size=3, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(1024, 1024, kernel_size=3, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(1024, 1024, kernel_size=3, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                
                nn.Conv1d(1024, 512, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                )
        self.fc = nn.Sequential(
                nn.Linear(512, 1))
    
    def forward(self, x):
        out =self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



class IMP:
    def __init__(self):
        self.model = Cnn(1, 1)
        self.model = torch.load('model')
        self.inputFasta = 'data/1kqf_C.fasta'
        self.input = 'data/1kqf_C.txt'
        self.finalResultPath ='result/'
        self.result = []
        self.features = []
        self.options = None
        
        
    def set_input(self, input):
        self.input = input
        self.inputFasta = input.split('.')[0] + '.fasta'
        
    def configure_option(self, option):
        self.options = option
        
    def read_input_file(self):
        # 
        f = open(self.input)
        lines = f.readlines()
        f.close()
        for line in lines:
            y = line.split(' ', 1)[0]
            tmpx = line.split(' ',1)[1]
            linex = tmpx.split(' ', 30)
            xAll = []
            for x in linex:
                xfeature = x.split(':', 1)[1]
                xAll.append(float(xfeature))
            self.features.append(xAll)
            
    def test_sample(self):
        # This function is used to test the test data.
        
        for each_sample in self.features:
            tmp = np.array(each_sample)
            tmp = tmp.reshape([1, 1, len(tmp)])
            x_test = torch.FloatTensor(tmp)
            x_test = Variable(x_test)
            out = self.model(x_test)
            predicted = np.float(out.data.numpy()[0,0])
            
            self.result.append(predicted)
            
    def write_to_file(self):
        if not self.result == None:
            
            f = open(self.inputFasta, 'r')
            seq = f.readline()
            seq = f.readline()
            f.close()
            seq = seq.replace("\n", "")
                            
            amino = len(seq)
            
            if not (amino*(amino-1)/2) == len(self.result):
                print('The fasta file can not provide correct number of amino acid name.')
                print('Could not write results to files.')
                return 
            
            count = 0
            strResult=''

            for x in range(amino+1):
                for y in range(amino+1):
                    if x == 0 and y == 0:
                        strResult = strResult + '%5.4s' %('')
                    elif x == 0 :
                        strResult = strResult + '%5.4s' %(seq[y-1])
                    elif y == 0 :
                        strResult = strResult + '%5.4s' %(seq[x-1])
                    elif y < x+1 :
                        strResult = strResult + '%5.4s' %('')
                    else:
                        strResult = strResult + '%5.4s' %(str(self.result[count]))
                        count = count + 1
                        
                strResult = strResult + '\n'
            f = open(self.finalResultPath + self.input.split('/')[-1].split('.')[0] + '_matrix', 'w')
            f.write(strResult)
            f.close()

            f = open(self.finalResultPath + self.input.split('/')[-1].split('.')[0] + '_list', 'w')
            count =0

            result = []
            for x in range(amino+1):
                for y in range(amino+1):
                    if not(x == 0 or y == 0  or y < x + 1):
                        tmp = x, seq[x-1], y, seq[y-1], self.result[count]
                        result.append(tmp)
                        count = count + 1

            result_sorted = sorted(result, key=lambda result:result[4], reverse=True)

            strResult='%5s' %('NO') + '|%8s' %('amino') + '|%5s' %('NO') + '|%8s' %('amino') + '|%12s' %('percentage')
            strResult = strResult + '\n'
            for i in result_sorted:
                strResult = strResult + '%5s' %(str(i[0])) + '|%8s' %(i[1]) + '|%5s' %(str(i[2])) + '|%8s' %(i[3]) + '|%12s' %(str(i[4]))
                strResult = strResult + '\n'
            print(strResult)
            f.write(strResult)
            f.close()
        
        
    def Run(self):
            
        if not self.options.input == None:
            if not os.path.exists(self.input):
                print('Error to locate the input name, please check the input file.')
            else:
                self.set_input(self.options.input)
        
            
        
        if not os.path.exists(self.input.split('.')[0] + '.fasta'):
            print('No fasta exist, please keep *.fasta file in current directory.')
            
        self.read_input_file()
        self.test_sample()
        self.write_to_file()
            
        
        
        