from mpi4py import MPI
import numpy as np
import time
import sys


def getSlices(comm, rank, numberOfCores, gridLegth):
    if rank == 0:
        lenthPerCore = 1.0 * gridLegth / numberOfCores
        begin = 1
        end = int(lenthPerCore)
        current = 1 + lenthPerCore

        for core in range(1, numberOfCores):
            indexes = (int(current), int(current + lenthPerCore - 1))
            comm.send(indexes, dest=core)
            current += lenthPerCore
    else:
        indexes = comm.recv(source=0)
        begin = indexes[0]
        end = indexes[1]
    return begin, end


def updateMatrix(comm, rank, numberOfCores, gridLegth, matrix, matrixTmp, upSlice, downSlice, begin, end):
    if rank % 2 == 0:
        if rank > 0:
            comm.Send([matrix[0], gridLegth + 2, MPI.DOUBLE], dest=rank - 1)
        if rank < numberOfCores - 1:
            comm.Recv(upSlice, source=rank + 1)
        if rank < numberOfCores - 1:
            comm.Send([matrix[end - begin], gridLegth + 2, MPI.DOUBLE], dest=rank + 1)
        if rank > 0:
            comm.Recv(downSlice, source=rank - 1)
    else:
        if rank < numberOfCores - 1:
            comm.Recv(upSlice, source=rank + 1)
        if rank > 0:
            comm.Send([matrix[0], gridLegth + 2, MPI.DOUBLE], dest=rank - 1)
        if rank > 0:
            comm.Recv(downSlice, source=rank - 1)
        if rank < numberOfCores - 1:
            comm.Send([matrix[end - begin], gridLegth + 2, MPI.DOUBLE], dest=rank + 1)

    for y in range(begin, end + 1):
        for x in range(1, gridLegth + 1):
            if y - 1 == 0:
                newDown = 0
            elif y - 1 >= begin:
                newDown = matrix[y - 1 - begin][x]
            else:
                newDown = downSlice[x]
            if y + 1 == gridLegth + 1:
                newUp = 0
            elif y + 1 <= end:
                newUp = matrix[y + 1 - begin][x]
            else:
                newUp = upSlice[x]
            matrixTmp[y - begin][x] = (matrix[y - begin][x - 1] + matrix[y - begin][x + 1] + newUp + newDown) / 4
    matrix = np.copy(matrixTmp)
    return matrix


def joinCalculations(comm, rank, numberOfCores, gridLegth, matrix, sliceLenth, result):
    if rank != 0:
        comm.Send([matrix, sliceLenth * (gridLegth + 2), MPI.DOUBLE], dest=0)
    else:
        index = 0

        for row in matrix:
            if row[1] == 0:
                break
            result[index] = row
            index += 1

        for i in range(numberOfCores - 1):
            temp = np.zeros((sliceLenth, gridLegth + 2), dtype=np.float64)
            comm.Recv(temp, source=i + 1)

            for x in temp:
                if x[1] == 0:
                    break

                result[index] = x
                index += 1


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numberOfCores = comm.Get_size()

    gridLegth = int(sys.argv[1])
    iterations = 100
    sliceLenth = int(gridLegth / numberOfCores + 0.5) + 1

    if rank == 0:
        times = 0

    for _ in range(numberOfTests):

        if rank == 0:
            start = time.time()

        begin, end = getSlices(comm, rank, numberOfCores, gridLegth)

        matrix = np.random.rand(sliceLenth, gridLegth + 2)

        for y in range(end - begin + 1):
            matrix[y][0] = 0
            matrix[y][gridLegth + 1] = 0

        matrixTmp = np.zeros((sliceLenth, gridLegth + 2), dtype=np.float64)
        upSlice = np.zeros(gridLegth + 2, dtype=np.float64)
        downSlice = np.zeros(gridLegth + 2, dtype=np.float64)
        result = np.zeros((gridLegth, gridLegth + 2), dtype=np.float64)

        for i in range(iterations):
            matrix = updateMatrix(comm, rank, numberOfCores, gridLegth, matrix, matrixTmp, upSlice, downSlice, begin, end)

        joinCalculations(comm, rank, numberOfCores, gridLegth, matrix, sliceLenth, result)

        if rank == 0:
            times += time.time() - start
            # print(result)

    if rank == 0:
        print("Took: ", times / numberOfTests)


if __name__ == '__main__':
    numberOfTests = 1
    main()
