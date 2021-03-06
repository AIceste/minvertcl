minvertcl: minvertcl.o clcooker.o
	gcc -g minvertcl.o clcooker.o -o minvertcl -lOpenCL -lm

minvertcl.o: minvertcl.c clcooker.h reduction_width.h error_code.h
	gcc -c -g -Wall -Wextra -Wpedantic -std=gnu11 -DCL_TARGET_OPENCL_VERSION=300 minvertcl.c

clcooker.o: clcooker.c clcooker.h
	gcc -c -g -Wall -Wextra -Wpedantic -std=gnu11 -DCL_TARGET_OPENCL_VERSION=300 clcooker.c

clean:
	rm -rf *.o minvertcl
