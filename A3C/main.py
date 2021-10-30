import A3C_discrete
import A3C_continuous


method = 0 # discrete
method = 1 # continuous

ONTRAIN = False
if __name__ == "__main__":
    if method:
        if ONTRAIN:
            A3C_continuous.train()
        else:
            A3C_continuous.test()
    else:
        if ONTRAIN:
            A3C_discrete.train()
        else:
            A3C_discrete.test()