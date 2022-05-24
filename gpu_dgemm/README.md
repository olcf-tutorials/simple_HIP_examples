This program fills an NxN matrix (where N is even), A, with alternating sin(index) and
cos(index) down its diagonal, performs the matrix multiply A*A on the GPU, then checks if
the sum of the diagonal of the resulting matrix equals N/2 since

    sin(index)*sin(index) + cos(index)*cos(index) = 1

If successful, the program output will be __SUCCESS__.
