#!/usr/bin/env python

from mpi4py import MPI
import signal


def handler(signum, frame):
    print('Signal handler called with signal', signum)


for s in (signal.SIGINT, signal.SIGTERM, signal.SIGSEGV, signal.SIGUSR1, signal.SIGUSR2):
    print(s)
    signal.signal(s, handler)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm.Set_errhandler(MPI.ERRORS_RETURN)

comm.Barrier()
if rank >= 2:
    import ctypes;ctypes.string_at(0)
comm.Barrier()

comm.Barrier()
