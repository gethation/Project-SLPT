from tqdm.auto import tqdm
import time
a = [i for i in range(1000)]

for i in tqdm(a):
    time.sleep(0.1)