import main_funks as m
from random import uniform
from config import settings as s

def main():
    delta = 0.05
    a_arr = m.make_heterogeneity_a(s.a, delta)
    

    print(a_arr)
if __name__ == '__main__':
    main()