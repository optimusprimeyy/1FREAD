from scipy.io import loadmat
import FREAD

Example = loadmat('FREAD_Example.mat')
demo_data = Example['trandata']

delta = 0.5
print('delta = ', delta)
score = FREAD.FREAD(demo_data, delta)
print('score = \n', score)
