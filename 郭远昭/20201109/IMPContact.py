'''
    This program is used to predict Binding sites of inner membrane proteins
    
    Usage:
        
        just run this program in python type the command like 
        
        ' python IMPContact '
        
        Program will run the demo program. amino name is 1kqf_C.
        
    Input:
        
        if you do not input any file. Program will predict the demo program automatically.
        
        The command i:
            
        ' python IMPContact '  or ' python IMPContact test '
        
        If there is another input file:
            
        ' python IMPContact -i ****.txt '
        
        * THe input file include two files. one is the feature file. it contains one label at first column,
            which indicate the whether two amino acid will react. 31 features attach to the label. for example
            
            
        '0 1:-1.11059595498 2:-1 3:-2 4:-1 5:-1 6:-2 7:-1 8:1.00 9:1 10:0 11:0.33 12:1 13:0 14:0.33 15:1 
            16:0 17:-1 18:-2 19:-1 20:1.00 21:1 22:0 23:0.33 24:1 25:0 26:0.33 27:1 28:0 29:0.33 30:1 31:0'
            
        * the other file is animo acid fasta file, which is the list of all animo acid. it should be mentioned
            that if the fasta file could not provide correct amino acid infomation(include more or less). The result
            file could not be written in file system.
       
    Output:
        
        There are two type of output files will generate. if predicted correctly. One with postfix matrix.
            it will organize as two dimension matrix. the row and the column are all amino acid list.
            
        the other with postfix list sorted by predicted value in desending order. 
        
        
    If there is any quesiton or advice or bug submit, please contact us. fangc447@nenu.edu.cn, wangh101@nenu.edu.cn.


'''
from optparse import OptionParser
from IMP import *
"""
师姐使用pytorch的实验代码，刚好最近需要学习
"""

def main():
    usage = ''
    parser = OptionParser(usage)
    parser.add_option('-t', '--test', dest='test', help='Test the test sample data.')
    parser.add_option('-i', '--input', dest='input', help='Specific the input file.')
    (options, args) = parser.parse_args()
    imp = IMP()
    imp.configure_option(options)
    imp.Run()


if __name__ == '__main__':
    main()