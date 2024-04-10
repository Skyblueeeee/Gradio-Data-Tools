import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

for each in os.listdir("log/pid"):
    os.system(f'taskkill -F -T -pid {each}')
    os.remove(f"log/pid/{each}")