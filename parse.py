import re
import  matplotlib.pyplot as plt

acc = 'val_acc: (.*)\n'

f = open('cnn_output.txt', 'r')
contents = f.read()

acclist = re.findall(acc, contents)

plt.plot(acclist)
plt.ylabel('test accuarcy')
plt.xlabel('epoch')
plt.show()