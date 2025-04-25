#include <iostream>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check for rowSize argument
    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <rowSize>\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rowSize = std::atoi(argv[1]);
    if (rowSize <= 0 || size % rowSize != 0) {
        if (rank == 0)
            std::cerr << "Error: rowSize must be >0 and divide total processes (" 
                      << size << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Compute 2D grid coords
    int myVal    = rank;
    int rowColor = rank / rowSize;   // which row
    int colColor = rank % rowSize;   // which column

    // Print (row, col) in rank order
    for (int p = 0; p < size; ++p) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            std::cout << "rank: " << rank
                      << "  row/col: (" << rowColor << "," << colColor << ")\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Create subcommunicators for rows and columns
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, rowColor, rank, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, colColor, rank, &colComm);

    // Get local ranks within subcomms
    int rowRank, colRank;
    MPI_Comm_rank(rowComm, &rowRank);
    MPI_Comm_rank(colComm, &colRank);

    // Reduce sums of global ranks within each subcomm
    int rowSum = 0, colSum = 0;
    MPI_Reduce(&myVal, &rowSum, 1, MPI_INT, MPI_SUM, 0, rowComm);
    MPI_Reduce(&myVal, &colSum, 1, MPI_INT, MPI_SUM, 0, colComm);

    // Print row‐sums in row order, from each rowComm root
    int nRows = size / rowSize;
    for (int r = 0; r < nRows; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rowColor == r && rowRank == 0) {
            std::cout << "Row " << r
                      << " root (global rank " << rank << "): sum = "
                      << rowSum << "\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Print col‐sums in column order, from each colComm root
    int nCols = rowSize;
    for (int c = 0; c < nCols; ++c) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (colColor == c && colRank == 0) {
            std::cout << "Col " << c
                      << " root (global rank " << rank << "): sum = "
                      << colSum << "\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Clean up
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Finalize();
    return 0;
}