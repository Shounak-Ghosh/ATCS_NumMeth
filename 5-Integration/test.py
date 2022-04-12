import numpy as np
import argparse


# main
def main(v,l):
    q = np.zeros(len(l))
    for i in range(len(l)):
        q[i] = int(l[i])
    print ("Q: ", q)
    # print("Hello World!")
    # a = np.arange(5, 10, 1)
    # a = np.linspace(5, 10, 7)
    # print(np.linspace(0, 5, 1))
    # print(np.linspace(0, 5, 2))
    # print(a[1:-1])
    # print(np.sum(a[1:-1]))
    # print(a)
    # print(a[:-1:2])


# run main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity", action="store_true")
    parser.add_argument('-l', '--list', nargs='+',
                        help='<Required> Set flag', required=True)
    args = parser.parse_args()

    main(args.verbose, args.list)
