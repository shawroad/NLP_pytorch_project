"""

@file  : test.py

@author: xiaolu

@time  : 2020-04-15

"""
import numpy as np

import math
import numpy as np

def clean_List_nan(List):
    Myarray=np.array(List)
    x = float('nan')
    for elem in Myarray:
        if math.isnan(x):
            x = 0.0
    return Myarray

nan = np.nan
oldlist =[nan, 19523.3211203121, 19738.4276377355, 19654.8478302742, 119.636737571360, 19712.4329437810, nan, 20052.3645613346, 19846.4815936009, 20041.8676619438, 19921.8126944154, nan, 20030.5073635719]

print(clean_List_nan(oldlist))

# if __name__ == '__main__':
#     data = [3, 23, 12, 432, 21, 42, 5]
#     data = np.array(data)
#     res = np.argsort(data)
#     res = res.tolist()
#     print(type(res))
#
#
#     print(res)
#     res = list(reversed(res))
#     print(res)
#
#
#     exit()
#

