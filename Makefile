minvertcl: minvertcl.o clcooker.o
	gcc -O3 minvertcl.o clcooker.o -o minvertcl -lOpenCL

minvertcl.o: minvertcl.c clcooker.h
	gcc -c -O3 -Wall -Wextra -Wpedantic -std=gnu11 -DCL_TARGET_OPENCL_VERSION=300 minvertcl.c

clcooker.o: clcooker.c clcooker.h
	gcc -c -O3 -Wall -Wextra -Wpedantic -std=gnu11 -DCL_TARGET_OPENCL_VERSION=300 clcooker.c

clean:
	rm -rf *.o minvertcl
